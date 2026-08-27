import re
import time
import tempfile
import shutil
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path


class EFetchRAG:
    """EFetch RAG tool for extracting context information about specific tables/figures from PubMed/PMC papers."""
    
    def __init__(self, context_depth: int = 1):
        self.context_depth = context_depth  # context-level, by default to extract only the first level, maximum 2
        self.temp_dir = None
        
        # PubMed API endpoints
        self.esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.elink_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        
        # Rate limiting
        self.request_delay = 0.5
        
        # Current BioC root for citation resolution
        self._current_bioc_root = None
        
    def _make_request_with_retry(self, url: str, params: dict = None, max_retries: int = 5) -> requests.Response:
        """Make HTTP request with retry logic for rate limits."""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    wait_time = (2 ** attempt) * self.request_delay
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2 ** attempt) * self.request_delay
                time.sleep(wait_time)
        
        raise requests.RequestException(f"Failed after {max_retries} retries")
    
    def _extract_id_from_input(self, input_str: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract PMID and PMC ID from various input formats."""
        pmid = None
        pmc_id = None
        
        # Try to extract PMID from URL or direct ID
        pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', input_str)
        if pmid_match:
            pmid = pmid_match.group(1)
        elif re.match(r'^\d+$', input_str):
            pmid = input_str
        elif 'id=' in input_str:
            id_match = re.search(r'id=(\d+)', input_str)
            if id_match:
                pmid = id_match.group(1)
        
        # Try to extract PMC ID
        pmc_match = re.search(r'(PMC\d+)', input_str)
        if pmc_match:
            pmc_id = pmc_match.group(1)
        
        return pmid, pmc_id
    
    def _get_pmc_id(self, pmid: str) -> Optional[str]:
        """Get PMC ID from PMID."""
        link_params = {
            'dbfrom': 'pubmed',
            'db': 'pmc',
            'id': pmid,
            'retmode': 'xml'
        }
        
        try:
            response = self._make_request_with_retry(self.elink_url, params=link_params)
            root = ET.fromstring(response.content)
            
            for linkset in root.findall('.//LinkSet'):
                for linksetdb in linkset.findall('.//LinkSetDb'):
                    dbto = linksetdb.find('.//DbTo')
                    if dbto is not None and dbto.text == 'pmc':
                        for link in linksetdb.findall('.//Link'):
                            id_elem = link.find('.//Id')
                            if id_elem is not None:
                                pmc_numeric = id_elem.text
                                return f"PMC{pmc_numeric}" if not pmc_numeric.startswith('PMC') else pmc_numeric
            
            return None
            
        except Exception as e:
            return None
    
    def _get_pmid_from_pmc(self, pmc_id: str) -> Optional[str]:
        """Get PMID from PMC ID."""
        link_params = {
            'dbfrom': 'pmc',
            'db': 'pubmed',
            'id': pmc_id,
            'retmode': 'xml'
        }
        
        try:
            response = self._make_request_with_retry(self.elink_url, params=link_params)
            root = ET.fromstring(response.content)
            
            for linkset in root.findall('.//LinkSet'):
                for linksetdb in linkset.findall('.//LinkSetDb'):
                    dbto = linksetdb.find('.//DbTo')
                    if dbto is not None and dbto.text == 'pubmed':
                        for link in linksetdb.findall('.//Link'):
                            id_elem = link.find('.//Id')
                            if id_elem is not None:
                                return id_elem.text
            
            return None
            
        except Exception as e:
            return None
    
    def _fetch_bioc_xml(self, pmid: str, pmc_id: str) -> Optional[ET.Element]:
        """Fetch BioC XML for the paper."""
        test_urls = [
            f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmc_id}/unicode",
            f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmid}/unicode"
        ]
        
        for url in test_urls:
            try:
                response = self._make_request_with_retry(url)
                text_content = response.text.strip()
                
                if any(err in text_content.lower()[:500] for err in ['error', 'not found', 'no document', 'invalid id']):
                    continue
                
                if not (text_content.startswith('<?xml') or text_content.startswith('<collection')):
                    continue
                
                try:
                    root = ET.fromstring(response.content)
                    
                    if root.tag in ['collection', 'document']:
                        passages = root.findall('.//passage')
                        if passages:
                            return root
                    
                except ET.ParseError:
                    continue
                
            except requests.RequestException:
                continue
        
        return None
    
    def _fetch_pubmed_metadata(self, pmid: str) -> Dict:
        """Fetch title, abstract, and references from PubMed."""
        fetch_params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml'
        }
        
        try:
            response = self._make_request_with_retry(self.efetch_url, params=fetch_params)
            xml_text = response.text
            root = ET.fromstring(response.content)
            
            # Extract title
            title_elem = root.find('.//ArticleTitle')
            title = title_elem.text.replace("\n", "").strip() if title_elem is not None else ""
            
            # Extract abstract XML
            abstract_pattern = r'(<Abstract>.*?</Abstract>)'
            abstract_match = re.search(abstract_pattern, xml_text, re.DOTALL)
            abstract = abstract_match.group(1) if abstract_match else ""
            
            # Extract references XML
            ref_pattern = r'(<ReferenceList>.*?</ReferenceList>)'
            ref_match = re.search(ref_pattern, xml_text, re.DOTALL)
            references = ref_match.group(1) if ref_match else ""
            
            return {
                'title': title,
                'abstract': abstract,
                'references': references
            }
            
        except Exception as e:
            return {
                'title': '',
                'abstract': '',
                'references': ''
            }
    
    def _parse_passage(self, passage: ET.Element) -> Dict:
        """Parse a passage element and extract all relevant information."""
        infons = {}
        for infon in passage.findall('.//infon'):
            key = infon.get('key', '')
            value = infon.text if infon.text is not None else ''
            infons[key] = value
        
        text_elem = passage.find('.//text')
        text_content = text_elem.text if text_elem is not None else ""
        
        offset_elem = passage.find('.//offset')
        offset = offset_elem.text if offset_elem is not None else ""
        
        return {
            'infons': infons,
            'type': infons.get('type', ''),
            'id': infons.get('id', ''),
            'section_type': infons.get('section_type', ''),
            'file': infons.get('file', ''),
            'text': text_content,
            'offset': offset,
            'xml': ET.tostring(passage, encoding='unicode')
        }
    
    def _find_table_in_bioc(self, table_data: str, bioc_root: ET.Element) -> Optional[Tuple[str, str, List[Dict]]]:
        """Find table in BioC XML by matching content."""
        passages = bioc_root.findall('.//passage')
        all_passages = [self._parse_passage(p) for p in passages]
        
        # Normalize input table content
        input_normalized = self._normalize_table_content(table_data)
        
        # Find all table passages
        table_passages = {}
        for passage_info in all_passages:
            if passage_info['type'] == 'table':
                table_id = passage_info['id']
                table_xml = passage_info['infons'].get('xml', '')
                
                if not table_xml:
                    continue
                
                table_normalized = self._normalize_table_content(table_xml)
                similarity = self._content_similarity(input_normalized, table_normalized)
                
                if similarity > 0.5:
                    table_passages[table_id] = {
                        'xml': table_xml,
                        'similarity': similarity,
                        'passage_info': passage_info
                    }
        
        if not table_passages:
            return None
        
        # Get the best match
        best_table_id = max(table_passages.keys(), key=lambda k: table_passages[k]['similarity'])
        best_match = table_passages[best_table_id]
        
        # Collect all passages related to this table
        matching_passages = [best_match['passage_info']]
        
        return best_table_id, best_match['xml'], matching_passages
    
    def _find_figure_in_bioc(self, figure_data: str, bioc_root: ET.Element) -> Optional[Tuple[str, str, List[Dict]]]:
        """Find figure in BioC XML by matching caption or file."""
        passages = bioc_root.findall('.//passage')
        all_passages = [self._parse_passage(p) for p in passages]
        
        # Try to extract figure ID or file from input
        input_file = None
        input_id = None
        input_caption = None
        
        # Parse input if it's XML
        try:
            input_root = ET.fromstring(figure_data)
            for infon in input_root.findall('.//infon'):
                key = infon.get('key', '')
                if key == 'file':
                    input_file = infon.text
                elif key == 'id':
                    input_id = infon.text
            text_elem = input_root.find('.//text')
            if text_elem is not None:
                input_caption = text_elem.text
        except:
            input_caption = figure_data
        
        # Find matching figure passages
        figure_passages = {}
        for passage_info in all_passages:
            if passage_info['type'] in ['fig_caption', 'figure_caption']:
                figure_id = passage_info['id']
                figure_file = passage_info['file']
                figure_caption = passage_info['text']
                
                # Match by ID
                if input_id and input_id == figure_id:
                    figure_passages[figure_id] = {
                        'xml': passage_info['xml'],
                        'score': 100,
                        'passage_info': passage_info
                    }
                    continue
                
                # Match by file
                if input_file and input_file == figure_file:
                    figure_passages[figure_id] = {
                        'xml': passage_info['xml'],
                        'score': 90,
                        'passage_info': passage_info
                    }
                    continue
                
                # Match by caption
                if input_caption and figure_caption:
                    if self._captions_match(input_caption, figure_caption):
                        figure_passages[figure_id] = {
                            'xml': passage_info['xml'],
                            'score': 80,
                            'passage_info': passage_info
                        }
        
        if not figure_passages:
            return None
        
        # Get the best match
        best_figure_id = max(figure_passages.keys(), key=lambda k: figure_passages[k]['score'])
        best_match = figure_passages[best_figure_id]
        
        matching_passages = [best_match['passage_info']]
        
        return best_figure_id, best_match['xml'], matching_passages
    
    def _normalize_table_content(self, content: str) -> str:
        """Normalize table content for matching."""
        content = re.sub(r'\s+', '', content)
        content = re.sub(r'%.*', '', content)
        content = content.lower()
        return content
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        if not content1 or not content2:
            return 0.0
        
        if content1 in content2:
            return len(content1) / len(content2)
        if content2 in content1:
            return len(content2) / len(content1)
        
        matches = 0
        i = j = 0
        while i < len(content1) and j < len(content2):
            if content1[i] == content2[j]:
                matches += 1
                i += 1
                j += 1
            else:
                found = False
                for k in range(j + 1, min(j + 10, len(content2))):
                    if content1[i] == content2[k]:
                        j = k + 1
                        matches += 1
                        i += 1
                        found = True
                        break
                if not found:
                    for k in range(i + 1, min(i + 10, len(content1))):
                        if content1[k] == content2[j]:
                            i = k + 1
                            matches += 1
                            j += 1
                            found = True
                            break
                if not found:
                    i += 1
                    j += 1
        
        avg_len = (len(content1) + len(content2)) / 2
        return matches / avg_len if avg_len > 0 else 0.0
    
    def _captions_match(self, caption1: str, caption2: str) -> bool:
        """Check if two captions match."""
        c1_words = set(caption1.lower().split())
        c2_words = set(caption2.lower().split())
        
        if len(c1_words) == 0 or len(c2_words) == 0:
            return False
        
        overlap = len(c1_words & c2_words)
        min_len = min(len(c1_words), len(c2_words))
        
        return overlap / min_len > 0.7
    
    def _extract_contexts_for_item(self, item_id: str, item_xml: str, all_passages: List[Dict], 
                                   item_type: str) -> List[Dict]:
        """Extract contexts (captions, paragraphs) for a table/figure."""
        contexts = []
        
        # Extract number from item_id (e.g., "table-1" -> 1)
        num_match = re.search(r'(\d+)', item_id)
        item_num = int(num_match.group(1)) if num_match else None
        
        for passage_info in all_passages:
            passage_type = passage_info['type']
            passage_id = passage_info['id']
            text_content = passage_info['text']
            section_type = passage_info['section_type']
            
            # Collect captions and footnotes for this item
            if item_type == 'table':
                if passage_id == item_id and passage_type in ['table_caption', 'table_footnote']:
                    contexts.append({
                        'content': text_content,
                        'type': passage_type,
                        'level': 1
                    })
            elif item_type == 'figure':
                if passage_id == item_id and passage_type in ['fig_caption', 'figure_caption']:
                    contexts.append({
                        'content': text_content,
                        'type': passage_type,
                        'level': 1
                    })
            
            # Find paragraphs that mention this item
            if passage_type == 'paragraph' and item_num:
                if self._text_mentions_item_by_number(text_content, item_type, item_num):
                    contexts.append({
                        'content': text_content,
                        'type': 'paragraph',
                        'level': 1
                    })
        
        # Extract citations from item XML
        citations_from_xml = self._extract_citations_from_text(item_xml)
        
        # Extract citations from contexts
        for context in contexts:
            citations_from_xml.update(self._extract_citations_from_text(context['content']))
        
        return contexts
    
    def _text_mentions_item_by_number(self, text: str, item_type: str, item_num: int) -> bool:
        """Check if text mentions an item by its sequential number."""
        if item_type == 'table':
            patterns = [
                rf'\bTable\s+{item_num}\b',
                rf'\bTab\.\s+{item_num}\b',
                rf'\bTable\s+{item_num}[AB]?\b',
                rf'\bSupplementary\s+Table\s+S?{item_num}\b'
            ]
        else:
            patterns = [
                rf'\bFig\.\s+{item_num}[AB]?\b',
                rf'\bFigure\s+{item_num}[AB]?\b',
                rf'\bFig\s+{item_num}[AB]?\b',
                rf'\bSupplementary\s+Fig\.\s+S?{item_num}[AB]?\b'
            ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _extract_citations_from_text(self, text: str) -> Set[str]:
        """Extract citation keys from text."""
        citations = set()
        
        # Pattern to match reference citations like [1], [2-4], [5,6,7]
        patterns = [
            r'\[(\d+)\]',
            r'\[(\d+)-(\d+)\]',
            r'\[(\d+),\s*(\d+)\]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    for num in match:
                        if num.isdigit():
                            citations.add(int(num))
                else:
                    if match.isdigit():
                        citations.add(int(match))
        
        return citations
    
    def _extract_second_level_contexts(self, first_level_contexts: List[Dict], 
                                      all_passages: List[Dict], original_id: str) -> List[Dict]:
        """Extract second-level contexts referenced by first-level contexts."""
        second_level_contexts = []
        seen_content = set()
        
        if original_id:
            seen_content.add(original_id)
        
        for context_item in first_level_contexts:
            content = context_item['content']
            
            # Extract references from this first-level context
            ref_pattern = r'\[(\d+)\]'
            ref_matches = re.finditer(ref_pattern, content)
            
            for ref_match in ref_matches:
                ref_num = ref_match.group(1)
                
                if ref_num in seen_content:
                    continue
                
                seen_content.add(ref_num)
                
                # Try to find referenced tables/figures
                for passage_info in all_passages:
                    passage_id = passage_info['id']
                    passage_type = passage_info['type']
                    
                    # Check if this is a table or figure with matching number
                    if passage_type == 'table':
                        num_match = re.search(r'(\d+)', passage_id)
                        if num_match and num_match.group(1) == ref_num:
                            xml_content = passage_info['infons'].get('xml', '')
                            if xml_content:
                                second_level_contexts.append({
                                    'content': xml_content,
                                    'type': 'table_content',
                                    'level': 2
                                })
                    
                    elif passage_type in ['fig_caption', 'figure_caption']:
                        num_match = re.search(r'(\d+)', passage_id)
                        if num_match and num_match.group(1) == ref_num:
                            second_level_contexts.append({
                                'content': passage_info['text'],
                                'type': 'figure_caption',
                                'level': 2
                            })
        
        return second_level_contexts
    
    def _remove_duplicate_contexts(self, contexts: List[Dict]) -> List[Dict]:
        """Remove duplicate contexts based on content similarity."""
        if not contexts:
            return contexts
        
        unique_contexts = []
        seen_normalized = set()
        
        for context_item in contexts:
            normalized = re.sub(r'\s+', ' ', context_item['content']).strip().lower()
            content_hash = hash(normalized[:500])
            
            if content_hash not in seen_normalized:
                seen_normalized.add(content_hash)
                unique_contexts.append(context_item)
            else:
                is_duplicate = False
                for existing in unique_contexts:
                    existing_normalized = re.sub(r'\s+', ' ', existing['content']).strip().lower()
                    if self._content_similarity(normalized, existing_normalized) > 0.9:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_contexts.append(context_item)
        
        return unique_contexts
    
    def extract_context(self, data: str, url_or_id: str) -> Dict:
        """Main method to extract context for a table/figure."""
        try:
            # Extract IDs from input
            pmid, pmc_id = self._extract_id_from_input(url_or_id)
            
            if not pmid and not pmc_id:
                return self.create_empty_result()
            
            # Get both IDs
            if pmid and not pmc_id:
                pmc_id = self._get_pmc_id(pmid)
            elif pmc_id and not pmid:
                pmid = self._get_pmid_from_pmc(pmc_id)
            
            if not pmid or not pmc_id:
                return self.create_empty_result()
            
            # Fetch BioC XML
            bioc_root = self._fetch_bioc_xml(pmid, pmc_id)
            if bioc_root is None:
                return self.create_empty_result()
            
            self._current_bioc_root = bioc_root
            
            # Fetch metadata
            metadata = self._fetch_pubmed_metadata(pmid)
            
            # Determine if input is table or figure
            is_table = '<table' in data.lower() or 'table' in data.lower()
            
            # Find the table/figure
            if is_table:
                result = self._find_table_in_bioc(data, bioc_root)
            else:
                result = self._find_figure_in_bioc(data, bioc_root)
            
            if not result:
                return self.create_empty_result()
            
            item_id, item_xml, matching_passages = result
            
            # Extract caption from matching passages
            caption = ""
            for passage_info in matching_passages:
                if passage_info['type'] in ['table_caption', 'fig_caption', 'figure_caption']:
                    caption = passage_info['text']
                    break
            
            # Get all passages
            passages = bioc_root.findall('.//passage')
            all_passages = [self._parse_passage(p) for p in passages]
            
            # Extract contexts
            item_type = 'table' if is_table else 'figure'
            contexts = self._extract_contexts_for_item(item_id, item_xml, all_passages, item_type)
            
            # Extract second-level contexts if needed
            if self.context_depth >= 2:
                second_level = self._extract_second_level_contexts(contexts, all_passages, item_id)
                contexts.extend(second_level)
            
            # Remove duplicates
            contexts = self._remove_duplicate_contexts(contexts)
            
            # Extract citations
            citations_nums = self._extract_citations_from_text(item_xml)
            for context in contexts:
                citations_nums.update(self._extract_citations_from_text(context['content']))
            
            # Build input_references and context_references
            input_references = []
            context_references = []
            
            # For now, just note the citation numbers
            for num in sorted(citations_nums):
                input_references.append(f"[{num}]")
            
            result_dict = {
                'caption': caption,
                'abstract': metadata['abstract'],
                'contexts': contexts,
                'input_references': input_references,
                'context_references': context_references,
                'label': item_id,
                'title': metadata['title'],
            }
            
            return result_dict
            
        except Exception as e:
            print(f"Error in extract_context: {e}")
            import traceback
            traceback.print_exc()
            return self.create_empty_result()
    
    def create_empty_result(self) -> Dict:
        """Create empty result dict."""
        return {
            'caption': '',
            'abstract': '',
            'contexts': [],
            'input_references': [],
            'context_references': [],
            'label': '',
            'title': '',
        }
    
    def format_contexts_for_agent(self, rag_result: dict):
        """Format contexts for agent consumption."""
        context_for_agent = [f"**ABSTRACT**:\n{rag_result['abstract']}"]
        
        for ctx_dict in rag_result["contexts"]:
            context_for_agent.append(f"**{ctx_dict['type'].upper()}**:\n{ctx_dict['content']}")
        
        ref_idx = 0
        for new_ref in rag_result["input_references"]:
            ref_idx += 1
            context_for_agent.append(f"**REFERENCE {ref_idx}**:\n{new_ref}")
        for new_ref in rag_result["context_references"]:
            ref_idx += 1
            context_for_agent.append(f"**REFERENCE {ref_idx}**:\n{new_ref}")
        
        context_for_agent_str = "* " + "\n* ".join(context_for_agent)
        return context_for_agent_str.strip()
