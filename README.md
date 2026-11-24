# 미니 포토샵

Windows에서 사진을 편집하고 AI로 피사체를 선택해 배경을 제거할 수 있는 데스크톱 이미지 편집기입니다. 여러 레이어와 문서 탭을 사용하고 작업 결과를 이미지 또는 PowerPoint로 내보낼 수 있습니다.

## 다운로드 및 실행

1. [최신 Windows 다운로드](https://github.com/parkwoobin/mini-photoshop/releases/latest)에서 `MiniPhotoshop.zip`을 받습니다.
2. 압축을 **모두 풀고** `MiniPhotoshop` 폴더의 `MiniPhotoshop.exe`를 실행합니다. 실행 파일과 `_internal` 폴더는 함께 있어야 합니다.
3. 파일 메뉴에서 사진을 열거나 이미지를 복사한 후 `Ctrl+V`로 붙여넣습니다.

배포판에는 Python, 라이브러리와 AI 모델이 포함되어 있습니다. Windows용으로 만들어졌으며 다른 운영체제는 검증되지 않았습니다.

## 주요 기능

- AI 피사체 선택: 사람이나 사물을 자동으로 선택하고 배경을 제거합니다. MODNet을 이용해 인물 가장자리를 다듬습니다.
- 수동 선택: 사각형, 올가미, 객체 선택, 빠른 선택과 자동 선택 도구를 제공합니다. Shift로 선택 영역을 추가하고 Alt로 제외할 수 있습니다.
- 레이어 편집: 이미지를 별도 레이어로 붙여넣고 이동, 크기 조절, 표시 및 삭제할 수 있습니다.
- 문서 작업: 여러 탭과 새 문서 프리셋, 이미지 및 캔버스 크기 변경, 확대/축소와 내비게이터를 지원합니다.
- 내보내기: PNG, JPEG, GIF와 PowerPoint를 지원합니다. PPT로 내보낼 때 보이는 레이어가 슬라이드의 별도 이미지 개체로 들어갑니다.
- 이미지 열기: JPEG, PNG, WebP, BMP, HEIC/HEIF를 지원합니다.

AI 선택 결과는 이미지와 모델에 따라 달라지며 작은 피사체나 머리카락 경계는 부정확할 수 있습니다. 필요하면 수동 선택 도구로 보정하세요. PPT의 이미지 개체에서 원본 텍스트를 편집할 수는 없습니다.

## 단축키

| 단축키 | 기능 |
| --- | --- |
| `Ctrl+N` | 새 문서 |
| `Ctrl+Shift+N` | 새 탭 |
| `Ctrl+C` / `Ctrl+V` | 선택 영역 또는 레이어 복사 / 붙여넣기 |
| `Ctrl+T` | 선택한 레이어 변형 (`Enter` 적용, `Esc` 취소) |
| `V` | 이동 도구 |
| `Delete` | 선택 영역 삭제 |
| `Ctrl+D` | 선택 해제 |
| `Ctrl+Z` / `Ctrl+Y` | 실행 취소 / 다시 실행 |
| `Ctrl+마우스 휠` | 확대 / 축소 |

## 사용 기술과 라이브러리

| 기술 | 용도 |
| --- | --- |
| Python | 애플리케이션 개발 언어 |
| PyQt5 / Qt | 데스크톱 UI, 캔버스와 사용자 입력 |
| Pillow, pillow-heif | 이미지 처리 및 HEIC/HEIF 지원 |
| NumPy | 이미지 배열과 수치 연산 |
| OpenCV | 이미지 처리와 선택 영역 계산 |
| ONNX Runtime | 로컬 AI 모델 추론 |
| YOLOv8 세그멘테이션 ONNX 모델 | 사람과 사물 감지 및 선택 |
| MODNet ONNX 모델 | 인물 가장자리 매팅 |
| python-pptx | PowerPoint 내보내기 |
| PyInstaller | Windows 실행 파일 패키징 |
| Lucide | UI 아이콘 |

정확한 Python 의존성 범위는 [requirements.txt](requirements.txt)를 참고하세요.

## 소스에서 실행

Windows에 Python을 설치한 뒤 PowerShell에서 프로젝트 폴더로 이동합니다. 배포판 ZIP의 `MiniPhotoshop/_internal/models` 폴더를 프로젝트 루트의 `models` 폴더로 복사하세요. 모델은 용량 때문에 소스 저장소에 포함하지 않았습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app_qt.py
```

또는 모델을 준비한 뒤 `실행.bat`을 실행하면 가상환경 및 의존성을 준비합니다. 최초 의존성 설치에는 인터넷 연결이 필요합니다.

## Windows 배포판 빌드

위와 같이 의존성과 `models`를 준비한 환경에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm MiniPhotoshop.spec
```

결과는 `dist/MiniPhotoshop`에 생성됩니다. 폴더 전체를 함께 배포하세요.

## 프로젝트 구성

```text
app_qt.py             메인 PyQt 애플리케이션
requirements.txt      실행 의존성
실행.bat              Windows 소스 실행 도우미
MiniPhotoshop.spec    PyInstaller 빌드 설정
pyi_preload_onnx.py   ONNX Runtime 로딩 훅
assets/               앱과 Lucide 아이콘
models/               AI 모델 (별도 다운로드)
```

## 모델과 라이선스

- YOLOv8 ONNX 모델 출처: [unileon-robotics/YOLO8-ONNX](https://huggingface.co/unileon-robotics/YOLO8-ONNX)
- MODNet 모델 출처: [Xenova/modnet](https://huggingface.co/Xenova/modnet)
- Lucide 아이콘: ISC 라이선스. 전문은 [assets/lucide/LICENSE](assets/lucide/LICENSE)를 참고하세요.
- 외부 모델과 라이브러리는 각각의 라이선스가 적용됩니다. 재배포나 상업적 이용 전 해당 프로젝트의 조건을 확인하세요.
- 이 프로젝트의 별도 라이선스는 아직 지정되지 않았습니다.
