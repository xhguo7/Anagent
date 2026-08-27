import os
import re
import tarfile
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple
import urllib.request
import shutil


class ArXivRAG:
    """ArXiv RAG tool for extracting context information about specific tables/figures from arXiv papers."""
    
    def __init__(self, context_depth: int = 1):
        self.temp_dir = None
        self.tex_files = []
        self.bib_files = []
        self.main_tex = None
        self.context_depth = context_depth  # context-level, by default to extract only the first level, maximum 2
        
    def download_arxiv_source(self, arxiv_url: str) -> str:
        """Download arXiv source package and extract it."""
        import time
        
        # Extract arXiv ID from URL
        arxiv_id = arxiv_url.split('/abs/')[-1]
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Download source tarball with retry logic
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        tar_path = os.path.join(self.temp_dir, "source.tar.gz")
        
        max_retries = 3
        download_success = False
        
        for attempt in range(max_retries):
            # Wait before each attempt (3s, 6s, 12s)
            wait_time = 3 * (2 ** attempt)
            print(f"Attempt {attempt + 1}/{max_retries}: Waiting {wait_time}s before download...")
            time.sleep(wait_time)
            
            try:
                urllib.request.urlretrieve(source_url, tar_path)
                
                # Check if we got a CAPTCHA page instead of the actual file
                with open(tar_path, 'rb') as f:
                    first_bytes = f.read(200)
                    if b'<html>' in first_bytes or b'reCAPTCHA' in first_bytes:
                        print(f"Got CAPTCHA on attempt {attempt + 1}, retrying...")
                        continue
                
                download_success = True
                break
                
            except Exception as e:
                print(f"Download failed on attempt {attempt + 1}: {e}")
                # Try without version number on last attempt
                if attempt == max_retries - 1:
                    arxiv_id_base = arxiv_id.split('v')[0]
                    source_url = f"https://arxiv.org/e-print/{arxiv_id_base}"
                    try:
                        time.sleep(3)
                        urllib.request.urlretrieve(source_url, tar_path)
                        
                        with open(tar_path, 'rb') as f:
                            first_bytes = f.read(200)
                            if b'<html>' not in first_bytes and b'reCAPTCHA' not in first_bytes:
                                download_success = True
                                break
                    except:
                        pass
        
        if not download_success:
            raise Exception(
                f"Failed to download arXiv source after {max_retries} attempts. "
                f"ArXiv may be rate-limiting requests. Please wait 10-15 minutes and try again. "
                f"URL: {arxiv_url}"
            )
        
        # Extract tarball
        extract_dir = os.path.join(self.temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        # Try different extraction methods
        try:
            # Method 1: Try as tar.gz with data filter (Python 3.12+)
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_dir, filter='data')
            return extract_dir
        except:
            pass
        
        try:
            # Method 2: Try as tar.gz without filter
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            return extract_dir
        except:
            pass
        
        try:
            # Method 3: Try as plain tar (no gzip)
            with tarfile.open(tar_path, 'r:') as tar:
                tar.extractall(extract_dir)
            return extract_dir
        except:
            pass
        
        try:
            # Method 4: Try as single gzip file
            import gzip
            with gzip.open(tar_path, 'rb') as f_in:
                tex_path = os.path.join(extract_dir, 'main.tex')
                with open(tex_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return extract_dir
        except:
            pass
        
        # Method 5: Treat as plain text file
        try:
            with open(tar_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(100)
                if '\\documentclass' in content or '\\begin{document}' in content or '\\section' in content:
                    tex_path = os.path.join(extract_dir, 'main.tex')
                    shutil.copy(tar_path, tex_path)
                    return extract_dir
        except:
            pass
        
        raise Exception(
            f"Could not extract arXiv source from {tar_path}. "
            f"Downloaded file successfully but all extraction methods failed. "
            f"File may be corrupted or in an unsupported format. "
            f"File size: {os.path.getsize(tar_path)} bytes"
        )

    def get_title_from_arxiv_api(self, arxiv_url: str) -> str:
        """Get title directly from arXiv API."""
        # Extract arXiv ID from URL
        arxiv_id = arxiv_url.split('/abs/')[-1]
        # Remove version number for API query
        arxiv_id_base = arxiv_id.split('v')[0]
        
        # Query arXiv API
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id_base}"
        
        try:
            with urllib.request.urlopen(api_url) as response:
                xml_data = response.read().decode('utf-8')
            
            # Parse XML
            root = ET.fromstring(xml_data)
            
            # Find title in the XML (namespace handling)
            namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
            title_elem = root.find('.//atom:entry/atom:title', namespaces)
            
            if title_elem is not None and title_elem.text:
                # Clean up the title text
                title = title_elem.text.strip()
                # Remove extra whitespace and newlines
                title = re.sub(r'\s+', ' ', title)
                return title
        except Exception as e:
            pass
        
        return ""

    def get_abstract_from_arxiv_api(self, arxiv_url: str) -> str:
        """Get abstract directly from arXiv API."""
        import urllib.request
        import xml.etree.ElementTree as ET
        
        # Extract arXiv ID from URL
        arxiv_id = arxiv_url.split('/abs/')[-1]
        # Remove version number for API query
        arxiv_id_base = arxiv_id.split('v')[0]
        
        # Query arXiv API
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id_base}"
        
        try:
            with urllib.request.urlopen(api_url) as response:
                xml_data = response.read().decode('utf-8')
            
            # Parse XML
            root = ET.fromstring(xml_data)
            
            # Find abstract in the XML (namespace handling)
            namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
            abstract_elem = root.find('.//atom:entry/atom:summary', namespaces)
            
            if abstract_elem is not None and abstract_elem.text:
                # Clean up the abstract text
                abstract = abstract_elem.text.strip()
                # Remove extra whitespace
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract
        except Exception as e:
            pass
        
        return ""
    
    def find_tex_and_bib_files(self, extract_dir: str):
        """Find all .tex and .bib files in extracted directory."""
        self.tex_files = list(Path(extract_dir).rglob("*.tex"))
        self.bib_files = list(Path(extract_dir).rglob("*.bib"))
        
        # Find main tex file (usually the one with \documentclass)
        for tex_file in self.tex_files:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if '\\documentclass' in content or '\\begin{document}' in content:
                    self.main_tex = tex_file
                    break
        
        if not self.main_tex and self.tex_files:
            self.main_tex = self.tex_files[0]
    
    def read_file_content(self, file_path: Path) -> str:
        """Read file content with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except:
            return ""
    
    def extract_caption_from_data(self, data: str) -> str:
        """Extract caption text from table/figure data, handling nested braces."""
        caption_match = re.search(r'\\caption\s*\{', data, re.DOTALL)
        if not caption_match:
            return ""
        
        # Find the starting position of the caption content
        start_pos = caption_match.end()
        
        # Now we need to find the matching closing brace, accounting for nested braces
        brace_count = 1
        pos = start_pos
        caption_end = -1
        
        while pos < len(data) and brace_count > 0:
            if data[pos] == '{' and (pos == 0 or data[pos-1] != '\\'):
                brace_count += 1
            elif data[pos] == '}' and (pos == 0 or data[pos-1] != '\\'):
                brace_count -= 1
                if brace_count == 0:
                    caption_end = pos
                    break
            pos += 1
        
        if caption_end == -1:
            return ""
        
        caption = data[start_pos:caption_end]
        
        # Clean up label inside caption
        caption = re.sub(r'\\label\{[^}]*\}', '', caption)
        
        return caption.strip()
    
    def extract_label_from_data(self, data: str) -> str:
        """Extract label from table/figure data."""
        label_match = re.search(r'\\label\{(.*?)\}', data)
        if label_match:
            return label_match.group(1)
        return ""
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison by removing extra whitespace and latex commands."""
        # Remove comments
        text = re.sub(r'%.*?\n', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common latex commands but keep text
        text = re.sub(r'\\(?:textbf|textit|emph|cite|ref|label)\{([^}]*)\}', r'\1', text)
        return text.strip()
    
    def extract_tabular_content(self, data: str) -> str:
        """Extract the core tabular content from input data."""
        # Find tabular environment
        tabular_match = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', data, re.DOTALL)
        if tabular_match:
            return tabular_match.group(0)
        
        # If no tabular environment, might be inside table*
        tabular_match = re.search(r'\\begin\{tabular\*\}.*?\\end\{tabular\*\}', data, re.DOTALL)
        if tabular_match:
            return tabular_match.group(0)
        
        # Might be tabularx or other variants
        for env in ['tabularx', 'tabulary', 'longtable']:
            tabular_match = re.search(rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}', data, re.DOTALL)
            if tabular_match:
                return tabular_match.group(0)
        
        return data
    
    def normalize_tabular_content(self, content: str) -> str:
        """Normalize tabular content for matching."""
        # Remove all whitespace and newlines
        content = re.sub(r'\s+', '', content)
        # Remove comments
        content = re.sub(r'%.*', '', content)
        # Normalize escaped underscores
        content = content.replace('\\_', '_')
        # Remove column specifications - they often differ
        content = re.sub(r'\\begin\{tabular\*?\}\{[^}]*\}', '\\begin{tabular}', content)
        content = re.sub(r'\\begin\{tabularx\}\{[^}]*\}\{[^}]*\}', '\\begin{tabular}', content)
        
        return content.lower()
    
    def extract_figure_content(self, data: str) -> str:
        """Extract the core figure content (includegraphics) from input data."""
        # Find includegraphics
        graphics_match = re.search(r'\\includegraphics.*?\{([^}]+)\}', data)
        if graphics_match:
            return graphics_match.group(1)
        return ""
    
    def find_table_or_figure_in_tex(self, data: str, tex_content: str) -> Optional[Tuple[str, str]]:
        """Find table/figure in tex content by matching caption or content."""
        # First try to match by caption or label if they exist in the input
        input_caption = self.extract_caption_from_data(data)
        input_label = self.extract_label_from_data(data)
        
        # Determine if it's a table or figure based on input data
        is_table = ('\\begin{tabular}' in data or '\\begin{table}' in data or 
                   '\\begin{tabularx}' in data or '\\begin{longtable}' in data)
        env_type = 'table' if is_table else 'figure'
        
        # Find all table/figure environments
        pattern = rf'\\begin\{{{env_type}\*?\}}.*?\\end\{{{env_type}\*?\}}'
        matches = list(re.finditer(pattern, tex_content, re.DOTALL | re.IGNORECASE))
        
        # Strategy 1: Match by label if available in input
        if input_label:
            for match in matches:
                env_text = match.group(0)
                if f'\\label{{{input_label}}}' in env_text:
                    label_match = re.search(r'\\label\{(.*?)\}', env_text)
                    label = label_match.group(1) if label_match else input_label
                    return label, env_text
        
        # Strategy 2: Match by caption if available in input
        if input_caption:
            input_caption_normalized = self.normalize_text(input_caption)
            
            for match in matches:
                env_text = match.group(0)
                env_caption = self.extract_caption_from_data(env_text)
                
                if env_caption:
                    env_caption_normalized = self.normalize_text(env_caption)
                    if self.captions_match(input_caption_normalized, env_caption_normalized):
                        label_match = re.search(r'\\label\{(.*?)\}', env_text)
                        label = label_match.group(1) if label_match else ""
                        return label, env_text
        
        # Strategy 3: Match by content (critical for inputs that only contain inner content)
        if is_table:
            input_tabular = self.extract_tabular_content(data)
            input_normalized = self.normalize_tabular_content(input_tabular)
            
            best_match = None
            best_score = 0.0
            
            for match in matches:
                env_text = match.group(0)
                env_tabular = self.extract_tabular_content(env_text)
                env_normalized = self.normalize_tabular_content(env_tabular)
                
                # Calculate content similarity
                score = self.content_similarity(input_normalized, env_normalized)
                
                if score > best_score and score > 0.5:  # Need at least 50% match
                    best_score = score
                    best_match = env_text
            
            if best_match:
                label_match = re.search(r'\\label\{(.*?)\}', best_match)
                label = label_match.group(1) if label_match else ""
                return label, best_match
        
        # Strategy 4: Match by figure filename
        if not is_table:
            input_filename = self.extract_figure_content(data)
            if input_filename:
                # Remove path and extension for matching
                input_base = os.path.splitext(os.path.basename(input_filename))[0]
                
                for match in matches:
                    env_text = match.group(0)
                    env_filename = self.extract_figure_content(env_text)
                    if env_filename:
                        env_base = os.path.splitext(os.path.basename(env_filename))[0]
                        if input_base.lower() == env_base.lower():
                            label_match = re.search(r'\\label\{(.*?)\}', env_text)
                            label = label_match.group(1) if label_match else ""
                            return label, env_text
        
        return None
    
    def content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings using character-level matching."""
        if not content1 or not content2:
            return 0.0
        
        # Check if one is substring of another (after normalization)
        if content1 in content2:
            return len(content1) / len(content2)
        if content2 in content1:
            return len(content2) / len(content1)
        
        # Use a simple character-level similarity metric
        matches = 0
        i = j = 0
        while i < len(content1) and j < len(content2):
            if content1[i] == content2[j]:
                matches += 1
                i += 1
                j += 1
            else:
                # Try to find the next match
                found = False
                # Look ahead in content2
                for k in range(j + 1, min(j + 10, len(content2))):
                    if content1[i] == content2[k]:
                        j = k + 1
                        matches += 1
                        i += 1
                        found = True
                        break
                if not found:
                    # Look ahead in content1
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
        
        # Calculate similarity as ratio of matches to average length
        avg_len = (len(content1) + len(content2)) / 2
        return matches / avg_len if avg_len > 0 else 0.0
    
    def captions_match(self, caption1: str, caption2: str) -> bool:
        """Check if two captions match (with some tolerance)."""
        c1_words = set(caption1.lower().split())
        c2_words = set(caption2.lower().split())
        
        if len(c1_words) == 0 or len(c2_words) == 0:
            return False
        
        # Calculate overlap
        overlap = len(c1_words & c2_words)
        min_len = min(len(c1_words), len(c2_words))
        
        # If more than 70% of words match, consider it a match
        return overlap / min_len > 0.7
    
    def extract_abstract(self, tex_content: str) -> str:
        """Extract abstract from tex content."""
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', 
                                   tex_content, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            return abstract_match.group(1).strip()
        return ""

    def extract_abstract_from_all_files(self, tex_content: str, all_tex_content: str, arxiv_url: str) -> str:
        """Extract abstract, trying arXiv API first, then tex files."""
        # Try arXiv API first (most reliable)
        abstract = self.get_abstract_from_arxiv_api(arxiv_url)
        if abstract:
            return abstract
        
        # Fallback: Try main tex file
        abstract = self.extract_abstract(tex_content)
        if abstract:
            return abstract
        
        # Last resort: Try all tex files combined
        abstract = self.extract_abstract(all_tex_content)
        return abstract
    
    def find_references_to_label(self, label: str, tex_content: str) -> List[str]:
        """Find all paragraphs that reference the given label."""
        if not label:
            return []
        
        paragraphs = []
        
        # Split into paragraphs (separated by blank lines)
        text_paragraphs = re.split(r'\n\s*\n', tex_content)
        
        # Look for references to this label
        ref_pattern = rf'\\(?:ref|autoref|cref|Cref|eqref)\{{[^}}]*{re.escape(label)}[^}}]*\}}'
        
        for para in text_paragraphs:
            if re.search(ref_pattern, para):
                # Clean paragraph
                cleaned = self.clean_paragraph(para)
                if cleaned:
                    paragraphs.append(cleaned)
        
        return paragraphs
    
    def find_references_to_table_figure_nearby(self, env_text: str, tex_content: str, env_type: str) -> List[str]:
        """Find paragraphs that reference the table/figure by searching nearby text."""
        paragraphs = []
        
        # Find the position of this environment in the text
        env_pos = tex_content.find(env_text)
        if env_pos == -1:
            # Try to find by searching for a distinctive part
            distinctive_part = env_text[:min(200, len(env_text))]
            env_pos = tex_content.find(distinctive_part)
        
        if env_pos == -1:
            return []
        
        # Search for generic references like "this table", "the table", "above table", etc
        window_size = 3000  # characters before and after
        
        start_pos = max(0, env_pos - window_size)
        end_pos = min(len(tex_content), env_pos + len(env_text) + window_size)
        
        nearby_text = tex_content[start_pos:end_pos]
        
        # Split into paragraphs
        text_paragraphs = re.split(r'\n\s*\n', nearby_text)
        
        # Look for generic references
        if env_type == 'table':
            ref_patterns = [
                r'\b(?:this|the|above|following|shown)\s+table\b',
                r'\bTable~?\d*\b',
                r'\btable~?\d*\b',
            ]
        else:
            ref_patterns = [
                r'\b(?:this|the|above|following|shown)\s+figure\b',
                r'\bFigure~?\d*\b',
                r'\bfigure~?\d*\b',
                r'\bFig\.~?\d*\b',
            ]
        
        for para in text_paragraphs:
            for pattern in ref_patterns:
                if re.search(pattern, para, re.IGNORECASE):
                    cleaned = self.clean_paragraph(para)
                    if cleaned and cleaned not in paragraphs:
                        paragraphs.append(cleaned)
                    break
        
        return paragraphs
    
    def clean_paragraph(self, para: str) -> str:
        """Clean a paragraph by removing comments and extra whitespace."""
        # Remove comments
        para = re.sub(r'%.*?\n', '\n', para)
        # Remove section commands
        para = re.sub(r'\\(?:section|subsection|subsubsection|paragraph)\*?\{.*?\}', '', para)
        # Remove labels
        para = re.sub(r'\\label\{.*?\}', '', para)
        # Clean whitespace
        para = re.sub(r'\s+', ' ', para)
        para = para.strip()
        
        # Filter out very short paragraphs (likely not meaningful)
        if len(para) < 50:
            return ""
        
        return para

    def find_caption_references(self, label: str, all_tex_content: str) -> List[Dict]:
        """Find captions from other tables/figures that reference this label."""
        caption_contexts = []
        
        if not label:
            return caption_contexts
        
        # Find all table and figure environments
        for env_type in ['table', 'figure']:
            pattern = rf'\\begin\{{{env_type}\*?\}}.*?\\end\{{{env_type}\*?\}}'
            matches = re.finditer(pattern, all_tex_content, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                env_text = match.group(0)
                # Check if this environment references our label
                ref_pattern = rf'\\(?:ref|autoref|cref|Cref|eqref)\{{[^}}]*{re.escape(label)}[^}}]*\}}'
                if re.search(ref_pattern, env_text):
                    caption = self.extract_caption_from_data(env_text)
                    if caption:
                        caption_contexts.append({
                            'content': caption,
                            'type': f'{env_type}_caption',
                            'level': 1
                        })
        
        return caption_contexts
    
    def find_footnote_references(self, label: str, all_tex_content: str) -> List[Dict]:
        """Find footnotes that reference this label."""
        footnote_contexts = []
        
        if not label:
            return footnote_contexts
        
        # Find all footnotes
        footnote_pattern = r'\\footnote\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
        matches = re.finditer(footnote_pattern, all_tex_content, re.DOTALL)
        
        ref_pattern = rf'\\(?:ref|autoref|cref|Cref|eqref)\{{[^}}]*{re.escape(label)}[^}}]*\}}'
        
        for match in matches:
            footnote_text = match.group(1)
            if re.search(ref_pattern, footnote_text):
                # Clean the footnote
                cleaned = re.sub(r'\\label\{.*?\}', '', footnote_text)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                if cleaned and len(cleaned) > 10:
                    footnote_contexts.append({
                        'content': cleaned,
                        'type': 'footnote',
                        'level': 1
                    })
        
        return footnote_contexts

    def extract_second_level_contexts(self, first_level_contexts: List[Dict], 
                                     all_tex_content: str, original_label: str) -> List[Dict]:
        """Extract second-level contexts referenced by first-level contexts."""
        second_level_contexts = []
        seen_content = set()  # To avoid duplicates
        
        # Add original label to avoid circular references
        if original_label:
            seen_content.add(original_label)
        
        for context_item in first_level_contexts:
            content = context_item['content']
            
            # Extract all references from this first-level context
            ref_pattern = r'\\(?:ref|autoref|cref|Cref|eqref)\{([^}]+)\}'
            ref_matches = re.finditer(ref_pattern, content)
            
            for ref_match in ref_matches:
                ref_label = ref_match.group(1).strip()
                
                # Skip if already seen or if it's the original label
                if ref_label in seen_content or ref_label == original_label:
                    continue
                
                seen_content.add(ref_label)
                
                # Try to find the referenced content
                table_fig_contexts = self.find_referenced_table_figure(ref_label, all_tex_content)
                second_level_contexts.extend(table_fig_contexts)
                
                # 2. Check if it's a section
                section_context = self.find_referenced_section(ref_label, all_tex_content)
                if section_context:
                    second_level_contexts.append(section_context)
                
                # 3. Check if it's an equation
                equation_context = self.find_referenced_equation(ref_label, all_tex_content)
                if equation_context:
                    second_level_contexts.append(equation_context)
        
        return second_level_contexts
    
    def find_referenced_table_figure(self, label: str, all_tex_content: str) -> List[Dict]:
        """Find table or figure with the given label and return its caption."""
        contexts = []
        
        for env_type in ['table', 'figure']:
            pattern = rf'\\begin\{{{env_type}\*?\}}.*?\\end\{{{env_type}\*?\}}'
            matches = re.finditer(pattern, all_tex_content, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                env_text = match.group(0)
                if f'\\label{{{label}}}' in env_text:
                    caption = self.extract_caption_from_data(env_text)
                    if caption:
                        contexts.append({
                            'content': caption,
                            'type': f'{env_type}_caption',
                            'level': 2
                        })
                    # Also include the full environment if it's informative
                    if env_type == 'table' and len(env_text) < 2000:
                        contexts.append({
                            'content': env_text,
                            'type': 'table_content',
                            'level': 2
                        })
                    break
        
        return contexts
    
    def find_referenced_section(self, label: str, all_tex_content: str) -> Optional[Dict]:
        """Find section with the given label."""
        # Find section command with this label nearby
        section_pattern = r'\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]+)\}'
        sections = list(re.finditer(section_pattern, all_tex_content))
        
        for i, section_match in enumerate(sections):
            section_start = section_match.end()
            # Check next ~500 characters for the label
            search_region = all_tex_content[section_start:section_start + 500]
            
            if f'\\label{{{label}}}' in search_region:
                section_title = section_match.group(1)
                
                # Extract content until next section or reasonable limit
                if i + 1 < len(sections):
                    section_end = sections[i + 1].start()
                else:
                    section_end = min(section_start + 3000, len(all_tex_content))
                
                section_content = all_tex_content[section_start:section_end]
                section_content = self.clean_paragraph(section_content)
                
                if section_content:
                    return {
                        'content': f"Section: {section_title}\n{section_content}",
                        'type': 'section',
                        'level': 2
                    }
        
        return None
    
    def find_referenced_equation(self, label: str, all_tex_content: str) -> Optional[Dict]:
        """Find equation with the given label."""
        # Look for equation environments
        equation_envs = ['equation', 'align', 'eqnarray', 'gather', 'multline']
        
        for env in equation_envs:
            pattern = rf'\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}'
            matches = re.finditer(pattern, all_tex_content, re.DOTALL)
            
            for match in matches:
                env_text = match.group(0)
                if f'\\label{{{label}}}' in env_text:
                    # Clean up the equation
                    cleaned = re.sub(r'\\label\{.*?\}', '', env_text)
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    
                    return {
                        'content': cleaned,
                        'type': 'equation',
                        'level': 2
                    }
        
        return None

    def remove_duplicate_contexts(self, contexts: List[Dict]) -> List[Dict]:
        """Remove duplicate contexts based on content similarity."""
        if not contexts:
            return contexts
        
        unique_contexts = []
        seen_normalized = set()
        
        for context_item in contexts:
            # Normalize content for comparison
            normalized = self.normalize_text(context_item['content'])
            
            # Use a hash of normalized content to detect duplicates
            content_hash = hash(normalized[:500])  # Use first 500 chars for hash
            
            if content_hash not in seen_normalized:
                seen_normalized.add(content_hash)
                unique_contexts.append(context_item)
            else:
                # Check if it's truly duplicate or just hash collision
                is_duplicate = False
                for existing in unique_contexts:
                    existing_normalized = self.normalize_text(existing['content'])
                    if self.content_similarity(normalized, existing_normalized) > 0.9:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_contexts.append(context_item)
        
        return unique_contexts
    
    def extract_citations_from_text(self, text: str) -> Set[str]:
        """Extract citation keys from text."""
        citations = set()
        
        # Match various citation commands
        cite_patterns = [
            r'\\cite\{([^}]+)\}',
            r'\\citep\{([^}]+)\}',
            r'\\citet\{([^}]+)\}',
            r'\\citealp\{([^}]+)\}',
            r'\\citealt\{([^}]+)\}',
            r'\\citeyear\{([^}]+)\}',
            r'\\citeauthor\{([^}]+)\}',
        ]
        
        for pattern in cite_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                cite_keys = match.group(1).split(',')
                for key in cite_keys:
                    citations.add(key.strip())
        
        return citations
    
    def get_bib_entry(self, cite_key: str) -> Optional[str]:
        """Get bibliography entry for a citation key."""
        for bib_file in self.bib_files:
            content = self.read_file_content(bib_file)
            
            # Find bib entry for this key
            pattern = rf'@\w+\{{{re.escape(cite_key)}\s*,.*?\n\}}'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            
            if match:
                return match.group(0)
        
        # Also check for embedded bibliography in tex files
        for tex_file in self.tex_files:
            content = self.read_file_content(tex_file)
            
            # Look in \begin{thebibliography}
            bib_match = re.search(r'\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}', 
                                 content, re.DOTALL)
            if bib_match:
                bib_content = bib_match.group(1)
                item_pattern = rf'\\bibitem\{{{re.escape(cite_key)}\}}(.*?)(?=\\bibitem|\\end{{thebibliography}})'
                item_match = re.search(item_pattern, bib_content, re.DOTALL)
                if item_match:
                    return f"\\bibitem{{{cite_key}}}{item_match.group(1)}"
        
        return None
    
    def extract_context(self, data: str, arxiv_url: str) -> Dict:
        """Main method to extract context for a table/figure."""
        try:
            # Download and extract source
            extract_dir = self.download_arxiv_source(arxiv_url)
            self.find_tex_and_bib_files(extract_dir)
            
            if not self.main_tex:
                return self.create_empty_result()
            
            # Read main tex content
            tex_content = self.read_file_content(self.main_tex)
            
            # Also read and combine all tex files for complete search
            all_tex_content = tex_content
            for tex_file in self.tex_files:
                if tex_file != self.main_tex:
                    all_tex_content += "\n" + self.read_file_content(tex_file)
            
            # Find the table/figure
            result = self.find_table_or_figure_in_tex(data, all_tex_content)
            
            if not result:
                return self.create_empty_result()
            
            label, env_text = result
            
            # Extract caption from the FOUND environment (not from input data)
            caption = self.extract_caption_from_data(env_text)
            
            # Extract abstract (try arXiv API first, then tex files)
            abstract = self.extract_abstract_from_all_files(tex_content, all_tex_content, arxiv_url)
            
            # Find contexts referencing this table/figure (level 1)
            contexts = []
            
            # Extract paragraphs
            if label:
                paragraphs = self.find_references_to_label(label, all_tex_content)
                for para in paragraphs:
                    contexts.append({
                        'content': para,
                        'type': 'paragraph',
                        'level': 1
                    })
                
                # Extract captions from other tables/figures
                caption_contexts = self.find_caption_references(label, all_tex_content)
                contexts.extend(caption_contexts)
                
                # Extract footnotes
                footnote_contexts = self.find_footnote_references(label, all_tex_content)
                contexts.extend(footnote_contexts)
            else:
                # Fallback: search for nearby references when no label
                is_table = '\\begin{tabular}' in data or '\\begin{table}' in data
                env_type = 'table' if is_table else 'figure'
                paragraphs = self.find_references_to_table_figure_nearby(env_text, all_tex_content, env_type)
                for para in paragraphs:
                    contexts.append({
                        'content': para,
                        'type': 'paragraph',
                        'level': 1
                    })
            
            # Extract second-level contexts if context_depth >= 2
            if self.context_depth >= 2:
                second_level = self.extract_second_level_contexts(contexts, all_tex_content, label)
                contexts.extend(second_level)
            
            # Remove duplicate contexts
            contexts = self.remove_duplicate_contexts(contexts)
            
            # Extract citations from the input table/figure environment
            input_citations = self.extract_citations_from_text(env_text)
            
            # Extract citations from contexts
            context_citations = set()
            for context_item in contexts:
                context_citations.update(self.extract_citations_from_text(context_item['content']))
            
            # Get bibliography entries for input table/figure
            input_references = []
            for cite_key in input_citations:
                bib_entry = self.get_bib_entry(cite_key)
                if bib_entry:
                    input_references.append(bib_entry)
            
            # Get bibliography entries for contexts
            context_references = []
            for cite_key in context_citations:
                bib_entry = self.get_bib_entry(cite_key)
                if bib_entry:
                    context_references.append(bib_entry)
            
            # Get references from contexts
            context_references = []
            for cite_key in context_citations:
                bib_entry = self.get_bib_entry(cite_key)
                if bib_entry:
                    context_references.append(bib_entry)
            
            # Create result dict
            result_dict = {
                'caption': caption,
                'abstract': abstract,
                'contexts': contexts,
                'input_references': input_references,
                'context_references': context_references,
                'label': label,
                'title': self.get_title_from_arxiv_api(arxiv_url),
            }
            
            return result_dict
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
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
        # Abstract
        context_for_agent = [f"**ABSTRACT**:\n{rag_result['abstract']}"]

        # Contexts
        for ctx_dict in rag_result["contexts"]:
            context_for_agent.append(f"**{ctx_dict['type'].upper()}**:\n{ctx_dict['content']}")
            
        # References
        ref_idx = 0
        for new_ref in rag_result["input_references"]:
            ref_idx += 1
            context_for_agent.append(f"**REFERENCE {ref_idx}**:\n{new_ref}")
        for new_ref in rag_result["context_references"]:
            ref_idx += 1
            context_for_agent.append(f"**REFERENCE {ref_idx}**:\n{new_ref}")

        context_for_agent_str = "* " + "\n* ".join(context_for_agent)
        return context_for_agent_str.strip()
