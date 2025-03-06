import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from pydub import AudioSegment
from audio_splitter import split_audio_file, get_audio_duration

class AudioSplitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("오디오 파일 분할기")
        self.root.geometry("500x350")
        
        # 변수 초기화
        self.file_path = ""
        self.output_dir = ""
        self.interval = tk.StringVar(value="60")  # 기본값 60초
        
        # UI 구성
        self.create_widgets()
    
    def create_widgets(self):
        # 파일 선택 프레임
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=15, fill=tk.X, padx=20)
        
        tk.Label(file_frame, text="오디오 파일:").grid(row=0, column=0, sticky=tk.W)
        self.file_label = tk.Label(file_frame, text="선택된 파일 없음", width=40, anchor="w")
        self.file_label.grid(row=0, column=1, padx=10)
        
        tk.Button(file_frame, text="파일 선택", command=self.select_file).grid(row=0, column=2)
        
        # 출력 디렉토리 선택 프레임
        output_frame = tk.Frame(self.root)
        output_frame.pack(pady=10, fill=tk.X, padx=20)
        
        tk.Label(output_frame, text="출력 폴더:").grid(row=0, column=0, sticky=tk.W)
        self.output_label = tk.Label(output_frame, text="선택된 폴더 없음", width=40, anchor="w")
        self.output_label.grid(row=0, column=1, padx=10)
        
        tk.Button(output_frame, text="폴더 선택", command=self.select_output_dir).grid(row=0, column=2)
        
        # 간격 설정 프레임
        interval_frame = tk.Frame(self.root)
        interval_frame.pack(pady=10, fill=tk.X, padx=20)
        
        tk.Label(interval_frame, text="분할 간격(초):").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(interval_frame, textvariable=self.interval, width=10).grid(row=0, column=1, sticky=tk.W, padx=10)
        
        # 파일 정보 프레임
        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=10, fill=tk.X, padx=20)
        
        tk.Label(info_frame, text="파일 정보:").grid(row=0, column=0, sticky=tk.W)
        self.info_label = tk.Label(info_frame, text="-", width=40, anchor="w")
        self.info_label.grid(row=0, column=1, padx=10, sticky=tk.W)
        
        # 실행 버튼
        tk.Button(self.root, text="오디오 분할하기", command=self.start_splitting, 
                  bg="#4CAF50", fg="white", height=2, width=20).pack(pady=20)
        
        # 상태 표시
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack(pady=10)
        
        # 진행 상황 표시
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(self.root, variable=self.progress_var, 
                                    from_=0, to=100, orient=tk.HORIZONTAL, 
                                    state=tk.DISABLED, length=460)
        self.progress_bar.pack(padx=20, pady=5)
    
    def select_file(self):
        file_types = [
            ("오디오 파일", "*.mp3 *.wav *.ogg *.flac *.aac"),
            ("모든 파일", "*.*")
        ]
        selected_file = filedialog.askopenfilename(filetypes=file_types)
        
        if selected_file:
            self.file_path = selected_file
            file_name = os.path.basename(self.file_path)
            self.file_label.config(text=file_name)
            self.status_label.config(text="파일이 선택되었습니다.")
            
            # 파일 정보 표시
            try:
                duration = get_audio_duration(self.file_path)
                self.info_label.config(text=f"재생 시간: {duration:.2f}초")
            except Exception as e:
                self.info_label.config(text=f"정보 로드 실패: {str(e)}")
    
    def select_output_dir(self):
        selected_dir = filedialog.askdirectory(title="분할된 파일을 저장할 폴더 선택")
        
        if selected_dir:
            self.output_dir = selected_dir
            dir_name = os.path.basename(selected_dir) or selected_dir
            self.output_label.config(text=dir_name)
            self.status_label.config(text="출력 폴더가 선택되었습니다.")
    
    def start_splitting(self):
        if not self.file_path:
            messagebox.showerror("오류", "오디오 파일을 선택해주세요.")
            return
        
        if not self.output_dir:
            messagebox.showerror("오류", "출력 폴더를 선택해주세요.")
            return
        
        try:
            interval_seconds = int(self.interval.get())
            
            if interval_seconds <= 0:
                messagebox.showerror("오류", "분할 간격은 1초 이상이어야 합니다.")
                return
            
            # 진행 상황 초기화
            self.progress_var.set(0)
            self.progress_bar.config(state=tk.NORMAL)
            
            # 스레드에서 분할 작업 실행
            threading.Thread(target=self.run_splitting, args=(interval_seconds,)).start()
            
        except ValueError:
            messagebox.showerror("오류", "분할 간격은 숫자로 입력해주세요.")
    
    def run_splitting(self, interval_seconds):
        try:
            self.status_label.config(text="오디오 파일을 분할 중...")
            
            # 오디오 파일 정보 가져오기
            duration = get_audio_duration(self.file_path)
            num_segments = int(duration / interval_seconds) + (1 if duration % interval_seconds > 0 else 0)
            
            # 진행 상황 업데이트 함수
            def progress_callback(current, total):
                progress = (current / total) * 100
                self.progress_var.set(progress)
                self.status_label.config(text=f"분할 중... ({current}/{total})")
                self.root.update_idletasks()
            
            # 분할 작업 수행
            output_files = self.split_with_progress(self.file_path, self.output_dir, interval_seconds, progress_callback)
            
            # 완료 메시지
            self.status_label.config(text=f"완료! {len(output_files)}개의 파일이 생성되었습니다.")
            messagebox.showinfo("완료", f"오디오 파일이 {len(output_files)}개로 분할되었습니다.\n저장 위치: {self.output_dir}")
            
        except Exception as e:
            messagebox.showerror("오류", f"파일 분할 중 오류가 발생했습니다: {str(e)}")
            self.status_label.config(text="오류가 발생했습니다.")
        finally:
            self.progress_bar.config(state=tk.DISABLED)
    
    def split_with_progress(self, file_path, output_dir, interval_seconds, callback=None):
        """
        진행 상황을 보고하면서 오디오 파일을 분할하는 함수
        """
        # 입력 검증
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 파일 확장자 확인
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 오디오 파일 로드
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
        elif file_ext == '.wav':
            audio = AudioSegment.from_wav(file_path)
        elif file_ext == '.ogg':
            audio = AudioSegment.from_ogg(file_path)
        elif file_ext == '.flac':
            audio = AudioSegment.from_file(file_path, "flac")
        elif file_ext == '.aac':
            audio = AudioSegment.from_file(file_path, "aac")
        else:
            audio = AudioSegment.from_file(file_path)
        
        # 밀리초 단위로 변환
        interval_ms = interval_seconds * 1000
        
        # 분할 작업
        total_length = len(audio)
        num_segments = (total_length // interval_ms) + (1 if total_length % interval_ms > 0 else 0)
        
        output_files = []
        
        for i in range(num_segments):
            start_time = i * interval_ms
            end_time = min((i + 1) * interval_ms, total_length)
            
            segment = audio[start_time:end_time]
            
            # 파일명 생성 (001, 002, ... 형식)
            segment_filename = f"{file_name}_{i+1:03d}{file_ext}"
            segment_path = os.path.join(output_dir, segment_filename)
            
            # 파일 저장
            segment.export(segment_path, format=file_ext.replace('.', ''))
            output_files.append(segment_path)
            
            # 진행 상황 콜백
            if callback:
                callback(i + 1, num_segments)
        
        return output_files

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioSplitterGUI(root)
    root.mainloop() 