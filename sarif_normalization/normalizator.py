import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


class SASTPreprocessor:
    def __init__(self, sarif_path: str, source_root: str, snippet_lines: int = 5):
        """
        Args:
            sarif_path: SARIF 파일 경로
            source_root: 소스 코드 루트 디렉토리
            snippet_lines: ±N줄 (기본 5줄)
        """
        self.sarif_path = sarif_path
        self.source_root = Path(source_root)
        self.snippet_lines = snippet_lines
        self.sarif_data = None
        self.file_cache = {}  # 파일 내용 캐싱
        
    def load_sarif(self):
        """SARIF 파일 로드"""
        with open(self.sarif_path, 'r', encoding='utf-8') as f:
            self.sarif_data = json.load(f)
    
    def read_source_file(self, file_path: str) -> List[str]:
        """소스 파일 읽기 (캐싱)"""
        if file_path in self.file_cache:
            return self.file_cache[file_path]
        
        full_path = self.source_root / file_path
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.file_cache[file_path] = lines
                return lines
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")
            return []
    
    def extract_code_range(self, file_path: str, start_line: int, end_line: int) -> Tuple[str, int, int]:
        """
        지정된 라인 범위의 코드 추출
        Returns: (code_string, actual_start_line, actual_end_line)
        """
        lines = self.read_source_file(file_path)
        if not lines:
            return "", start_line, end_line
        
        # 1-based index를 0-based로 변환
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        code = ''.join(lines[start_idx:end_idx])
        return code, start_idx + 1, end_idx
    
    def extract_trigger_line(self, file_path: str, line_num: int) -> str:
        """
        Trigger 라인 추출 (원본 유지 - 탭/공백 정규화 안 함)
        우측 개행만 제거
        """
        lines = self.read_source_file(file_path)
        if not lines or line_num < 1 or line_num > len(lines):
            return ""
        
        return lines[line_num - 1].rstrip('\n\r')
    
    def find_function_boundary(self, file_path: str, line_num: int, language: str) -> Optional[Tuple[int, int, str]]:
        """
        함수 경계 찾기
        실패 시 None 반환 → snippet으로 fallback
        """
        lines = self.read_source_file(file_path)
        if not lines:
            return None
        
        if language.lower() == 'python':
            return self._find_function_python(lines, line_num)
        elif language.lower() in ['javascript', 'typescript', 'java', 'c', 'cpp', 'go']:
            return self._find_function_brace_based(lines, line_num, language)
        else:
            return None
    
    def _find_function_python(self, lines: List[str], line_num: int) -> Optional[Tuple[int, int, str]]:
        """Python: indent 기반 함수/메서드 경계 탐지"""
        # 함수/메서드 정의 패턴 (class method, async 포함)
        pattern = r'^\s*(?:async\s+)?def\s+(\w+)\s*\('
        
        # 함수 시작 찾기 (최대 100줄 위까지)
        func_start = None
        func_name = None
        func_indent = None
        
        for i in range(line_num - 1, max(-1, line_num - 100), -1):
            line = lines[i]
            match = re.match(pattern, line)
            if match:
                func_start = i + 1
                func_name = match.group(1)
                func_indent = len(line) - len(line.lstrip())
                break
        
        if not func_start:
            return None
        
        # indent 기반 종료 찾기
        func_end = func_start
        in_function = True
        
        for i in range(func_start, min(len(lines), func_start + 300)):
            line = lines[i]
            
            # 빈 줄이나 주석은 스킵
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                func_end = i + 1
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            # 함수 정의보다 작거나 같은 indent면 종료
            # 단, 바로 다음 줄이 아니어야 함
            if i > func_start and current_indent <= func_indent:
                # 다른 def나 class 만나면 확실히 종료
                if re.match(r'^\s*(?:def|class|async\s+def)\s+', line):
                    break
                # 같은 레벨의 코드면 종료
                if current_indent == func_indent:
                    break
            
            func_end = i + 1
        
        return func_start, func_end, func_name
    
    def _find_function_brace_based(self, lines: List[str], line_num: int, language: str) -> Optional[Tuple[int, int, str]]:
        """
        중괄호 기반 함수 경계 탐지 (JavaScript/C 계열)
        한계: class method, arrow function, 주석 내 중괄호 오탐 가능
        → 실패하면 None 반환하여 snippet으로 fallback
        """
        patterns = {
            'javascript': r'^\s*(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>)|(\w+)\s*\([^)]*\)\s*\{)',
            'typescript': r'^\s*(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>)|(\w+)\s*\([^)]*\)\s*\{)',
            'java': r'^\s*(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(',
            'c': r'^\s*(?:static\s+)?[\w\*]+\s+(\w+)\s*\(',
            'cpp': r'^\s*(?:static\s+)?[\w\*:<>]+\s+(\w+)\s*\(',
            'go': r'^\s*func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(',
        }
        
        pattern = patterns.get(language.lower())
        if not pattern:
            return None
        
        # 함수 시작 찾기 (최대 50줄 위까지)
        func_start = None
        func_name = None
        for i in range(line_num - 1, max(-1, line_num - 50), -1):
            match = re.match(pattern, lines[i])
            if match:
                func_start = i + 1
                func_name = next((g for g in match.groups() if g), 'anonymous')
                break
        
        if not func_start:
            return None
        
        # 중괄호 매칭 (간단한 카운터)
        brace_count = 0
        started = False
        func_end = func_start
        
        for i in range(func_start - 1, min(len(lines), func_start + 200)):
            line = lines[i]
            
            # 주석/문자열 무시는 하지 않음 (복잡도 증가)
            for char in line:
                if char == '{':
                    brace_count += 1
                    started = True
                elif char == '}':
                    brace_count -= 1
                    if started and brace_count == 0:
                        func_end = i + 1
                        return func_start, func_end, func_name
        
        # 중괄호 매칭 실패 → None
        return None
    
    def extract_snippet(self, file_path: str, line_num: int) -> Tuple[str, int, int]:
        """±N줄 스니펫 추출"""
        lines = self.read_source_file(file_path)
        if not lines:
            return "", line_num, line_num
        
        start = max(1, line_num - self.snippet_lines)
        end = min(len(lines), line_num + self.snippet_lines)
        
        return self.extract_code_range(file_path, start, end)
    
    def is_taint_mode(self, result) -> bool:
        """Taint 모드인지 확인 (codeFlows 존재 여부)"""
        return 'codeFlows' in result and len(result['codeFlows']) > 0
    
    def _infer_role(self, location: Dict, index: int, total: int) -> Tuple[str, str]:
        """
        Role 추론 with confidence level
        Returns: (role, confidence)
        confidence: 'explicit' > 'kinds-based' > 'index-based' > 'message-based'
        """
        props = location.get('properties', {})
        kinds = location.get('kinds', [])
        message = location.get('message', {}).get('text', '').lower()
        
        # 1. Explicit property (최고 신뢰도)
        if props.get('role'):
            return props['role'], 'explicit'
        
        # 2. Kinds 기반 (높은 신뢰도)
        if 'source' in kinds:
            return 'source', 'kinds-based'
        elif 'sink' in kinds:
            return 'sink', 'kinds-based'
        elif 'sanitizer' in kinds:
            return 'sanitizer', 'kinds-based'
        
        # 3. Index 기반 (중간 신뢰도)
        if index == 0:
            return 'source', 'index-based'
        elif index == total - 1:
            return 'sink', 'index-based'
        
        # 4. Message 기반 (낮은 신뢰도)
        if 'taint source' in message or 'user input' in message:
            return 'source', 'message-based'
        elif 'taint sink' in message or 'dangerous' in message:
            return 'sink', 'message-based'
        elif 'sanitizer' in message or 'sanitized' in message:
            return 'sanitizer', 'message-based'
        
        # 5. Fallback
        return 'transform', 'index-based'
    
    def extract_dataflow(self, result) -> List[Dict]:
        """Dataflow 추출 with confidence"""
        flows = []
        
        if 'codeFlows' not in result:
            return flows
        
        for code_flow in result['codeFlows']:
            for thread_flow in code_flow.get('threadFlows', []):
                locations = thread_flow.get('locations', [])
                
                for idx, location in enumerate(locations):
                    phys_loc = location.get('location', {}).get('physicalLocation', {})
                    file_path = phys_loc.get('artifactLocation', {}).get('uri', '')
                    region = phys_loc.get('region', {})
                    
                    start_line = region.get('startLine', 0)
                    end_line = region.get('endLine', start_line)
                    
                    # Role 추출 with confidence
                    role, confidence = self._infer_role(location, idx, len(locations))
                    
                    flows.append({
                        'file': file_path,
                        'start_line': start_line,
                        'end_line': end_line,
                        'role': role,
                        'role_confidence': confidence
                    })
        
        return flows
    
    def _compare_confidence(self, c1: str, c2: str) -> int:
        """
        신뢰도 비교
        Returns: -1 if c2 higher, 0 if same, 1 if c1 higher
        """
        order = ['explicit', 'kinds-based', 'index-based', 'message-based']
        idx1 = order.index(c1) if c1 in order else 99
        idx2 = order.index(c2) if c2 in order else 99
        return idx2 - idx1
    
    def deduplicate_code_units(self, code_units: List[Dict]) -> List[Dict]:
        """
        중복 제거 - 같은 파일+라인 범위는 통합
        roles는 array로 통일
        """
        seen = {}
        result = []
        
        for cu in code_units:
            key = (cu['file'], tuple(cu['line_range']))
            
            if key in seen:
                idx = seen[key]
                existing = result[idx]
                # role 병합
                if cu['role'] not in existing['roles']:
                    existing['roles'].append(cu['role'])
                # confidence는 가장 높은 것 유지
                if self._compare_confidence(cu['role_confidence'], existing['role_confidence']) > 0:
                    existing['role_confidence'] = cu['role_confidence']
            else:
                seen[key] = len(result)
                cu['roles'] = [cu['role']]
                del cu['role']
                result.append(cu)
        
        return result
    
    def process_result(self, result, rule_id: str, language: str) -> Dict:
        """개별 finding 처리"""
        is_taint = self.is_taint_mode(result)
        
        finding = {
            'rule_id': rule_id,
            'language': language,
            'is_taint': is_taint,
            'code_units': []
        }
        
        if is_taint:
            # Taint 모드: dataflow 추출
            dataflows = self.extract_dataflow(result)
            
            cu_counter = 1
            for df in dataflows:
                file_path = df['file']
                start_line = df['start_line']
                role = df['role']
                role_confidence = df['role_confidence']
                
                # 함수 경계 찾기
                func_info = self.find_function_boundary(file_path, start_line, language)
                
                if func_info:
                    func_start, func_end, func_name = func_info
                    scope_type = "function"
                    code, actual_start, actual_end = self.extract_code_range(file_path, func_start, func_end)
                else:
                    # 함수 찾기 실패 → 스니펫으로 fallback
                    scope_type = "global"
                    code, actual_start, actual_end = self.extract_snippet(file_path, start_line)
                
                # Trigger 라인 추출 (원본 유지)
                trigger_code = self.extract_trigger_line(file_path, start_line)
                
                finding['code_units'].append({
                    'id': f"cu_{cu_counter}",
                    'role': role,
                    'role_confidence': role_confidence,
                    'file': file_path,
                    'scope_type': scope_type,
                    'line_range': [actual_start, actual_end],
                    'trigger_line': start_line,
                    'trigger_code': trigger_code,
                    'full_code': code
                })
                cu_counter += 1
            
            # 중복 제거
            finding['code_units'] = self.deduplicate_code_units(finding['code_units'])
        
        else:
            # Non-taint 모드
            locations = result.get('locations', [])
            if not locations:
                return finding
            
            location = locations[0]
            phys_loc = location.get('physicalLocation', {})
            file_path = phys_loc.get('artifactLocation', {}).get('uri', '')
            region = phys_loc.get('region', {})
            
            start_line = region.get('startLine', 0)
            end_line = region.get('endLine', start_line)
            
            # 함수 안인지 확인
            func_info = self.find_function_boundary(file_path, start_line, language)
            
            if func_info:
                # 함수 내: 함수 전체
                func_start, func_end, func_name = func_info
                scope_type = "function"
                code, actual_start, actual_end = self.extract_code_range(file_path, func_start, func_end)
            else:
                # 함수 밖: ±5줄 스니펫
                scope_type = "global"
                code, actual_start, actual_end = self.extract_snippet(file_path, start_line)
            
            # Trigger 라인 추출
            trigger_code = self.extract_trigger_line(file_path, start_line)
            
            finding['code_units'].append({
                'id': 'cu_1',
                'roles': ['sink'],
                'role_confidence': 'pattern-based',  # Non-taint는 패턴 매칭
                'file': file_path,
                'scope_type': scope_type,
                'line_range': [actual_start, actual_end],
                'trigger_line': start_line,
                'trigger_code': trigger_code,
                'full_code': code
            })
        
        return finding
    
    def process(self) -> Dict:
        """전체 SARIF 파일 처리"""
        self.load_sarif()
        
        output = {
            'findings': []
        }
        
        for run in self.sarif_data.get('runs', []):
            # Rule 정보 추출
            rules = {}
            tool_driver = run.get('tool', {}).get('driver', {})
            for rule in tool_driver.get('rules', []):
                rule_id = rule.get('id')
                # 언어 정보는 properties나 tags에서 추출
                tags = rule.get('properties', {}).get('tags', [])
                language = 'unknown'
                for tag in tags:
                    if '.' in tag:  # javascript.express.security 같은 형태
                        language = tag.split('.')[0]
                        break
                rules[rule_id] = {'language': language}
            
            # Results 처리
            for result in run.get('results', []):
                rule_id = result.get('ruleId', 'unknown')
                
                # 언어 정보 추출: rule_id에서 먼저 시도
                language = 'unknown'
                if rule_id and '.' in rule_id:
                    potential_lang = rule_id.split('.')[0]
                    # 알려진 언어인지 확인
                    known_langs = ['python', 'javascript', 'typescript', 'java', 'go', 'c', 'cpp', 
                                   'ruby', 'php', 'rust', 'kotlin', 'swift', 'bash', 'yaml', 'json']
                    if potential_lang.lower() in known_langs:
                        language = potential_lang.lower()
                
                # rule_id에서 못 찾으면 rules 딕셔너리에서
                if language == 'unknown':
                    language = rules.get(rule_id, {}).get('language', 'unknown')
                
                finding = self.process_result(result, rule_id, language)
                if finding['code_units']:  # code_units가 있는 경우만 추가
                    output['findings'].append(finding)
        
        return output
    
    def save_output(self, output_path: str):
        """결과 JSON 저장"""
        output = self.process()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, indent=2, ensure_ascii=False, fp=f)
        print(f"✓ Preprocessed {len(output['findings'])} findings to {output_path}")


# 사용 예시
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SAST Preprocessor for SARIF files')
    parser.add_argument('sarif', help='SARIF file path')
    parser.add_argument('source_root', help='Source code root directory')
    parser.add_argument('-o', '--output', default='preprocessed_findings.json', 
                       help='Output JSON file path')
    parser.add_argument('-n', '--snippet-lines', type=int, default=5,
                       help='Number of lines for snippets (±N)')
    
    args = parser.parse_args()
    
    preprocessor = SASTPreprocessor(
        sarif_path=args.sarif,
        source_root=args.source_root,
        snippet_lines=args.snippet_lines
    )
    
    preprocessor.save_output(args.output)