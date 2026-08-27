# Cat Meme Detector

웹캠에서 사람 얼굴 표정과 손동작을 분석해 가장 어울리는 고양이 밈을 보여 주는
C++17/OpenCV 5.0 프로그램입니다. 안경을 쓴 얼굴도 일반 사람 얼굴로 검출하며,
손이 보이면 palm 검출 뒤 21개 관절을 추적합니다.

## 매칭 방식

- 얼굴: Haar Cascade로 가장 큰 사람 얼굴을 찾습니다.
- 표정: OpenCV Zoo MobileFaceNet 모델로 `angry`, `disgust`, `fearful`, `happy`,
  `neutral`, `sad`, `surprised`를 분석합니다.
- 손: MediaPipe palm 모델과 hand-pose 모델로 21개 관절을 얻고 `fist`,
  `open-palm`, `peace`, `pointing`, `thumbs-up`, `other`로 분류합니다.
- 밈: `assets/meme_labels.csv`의 authored 표정/손동작 라벨과 영상 특징을 함께 비교합니다.

손이 없을 때는 표정 85% + 영상 특징 15%, 손이 있을 때는 표정 45% + 손동작
45% + 영상 특징 10%로 최종 점수를 계산합니다. 얼굴이 검출되면 임계값보다 낮아도
가장 가까운 결과를 `CLOSEST`로 보여 주고, 임계값 이상이면 `MATCH`로 표시합니다.

## CPU 빌드

Windows에서 아래 스크립트가 OpenCV 5.0과 contrib 5.0을 내려받아 필요한 정적
라이브러리와 앱을 빌드하고 회귀 테스트를 실행합니다. 첫 빌드는 시간이 걸립니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\build.ps1
```

출력은 `build-opencv5\cat_meme_detector.exe`입니다.

## RTX 5060 Ti CUDA 빌드

RTX 5060 Ti는 이 스크립트에서 compute capability 12.0(`sm_120`)으로 빌드합니다.
먼저 Windows용 CUDA Toolkit 13.x와 호환되는 cuDNN 9를 설치하고 새 PowerShell을
여십시오. cuDNN을 사용자 경로에 설치했다면 `CUDNN_ROOT`를 그 디렉터리로 설정합니다.

```powershell
$env:CUDNN_ROOT = 'C:\path\to\cudnn'
.\scripts\build.ps1 -EnableCuda
```

CUDA 빌드는 기존 CPU 빌드를 건드리지 않고 `build-opencv5-cuda`와
`.deps\opencv-install-cuda`에 별도로 생성됩니다. 실행 시 CUDA FP16을 우선 사용하며,
GPU backend 초기화나 추론이 실패하면 자동으로 CPU로 전환합니다. 현재 backend는
콘솔, 화면 하단, `cat_meme_detector.log`에서 확인할 수 있습니다. 강제로 CPU를 쓰려면
`--cpu`를 지정합니다.

## 실행

```powershell
.\build-opencv5\cat_meme_detector.exe
```

주요 옵션:

```text
--camera 1           다른 웹캠 사용
--threshold 40       MATCH 표시 기준(0~100, 기본값 40)
--memes D:/memes     다른 밈 폴더 사용
--image portrait.jpg 한 장의 사진 분석
--cpu                CUDA를 사용하지 않고 CPU 추론
--check-assets       밈 전처리와 authored asset 검사 후 종료
```

종료는 `Q` 또는 `Esc`, 실행 중 밈 다시 읽기는 `R`입니다. 시작 즉시 콘솔에 얼굴 모델,
밈 분석, 표정·손 ONNX 모델의 progress bar가 표시됩니다. `R` 재로딩 때도 밈 바가
`0/N`부터 다시 표시됩니다. 화면에는 얼굴 영역, 손 박스와 21개 관절, 표정, 손동작,
DNN backend가 함께 표시됩니다.

밈도 시작 시 첫 프레임을 DNN으로 분석합니다. `meme_labels.csv`에 수동 라벨이 있으면
그 값을 우선하고, 라벨이 없으면 자동 분석 결과로 채웁니다. CSV의 손동작 `none`은
명시적인 손 없는 밈으로 유지됩니다. 따라서 새 밈은 CSV를 수정하지 않아도 바로 사용할
수 있고, 필요할 때만 수동 라벨로 결과를 보정하면 됩니다.

웹캠에서 손이 감지되지 않으면 `none` 라벨 밈이 하나라도 있는 한 pointing/fist 같은
손동작 밈은 후보에서 제외합니다. 기본 neutral 표정에서는 손 없는 `son.gif` 같은 밈으로
fallback합니다.

손동작은 단일 DNN 결과를 즉시 사용하지 않습니다. 5프레임 간격의 분석에서 같은 gesture가
4회 연속 확인돼야 활성화되고, 활성화 후 2회 연속 손을 놓쳐야 `none`으로 돌아갑니다.
얼굴이나 배경을 손으로 잘못 잡아 gesture가 빠르게 바뀌는 오탐은 매칭에 반영되지 않습니다.

## 밈 추가와 라벨

이미지를 `assets/memes` 아래에 넣으면 자동 분석됩니다. 자동 결과를 보정하고 싶을 때만
`assets/meme_labels.csv`에 다음 형식으로 한 줄을 추가합니다. JPG, PNG, WEBP, BMP,
GIF를 지원하며 GIF는 애니메이션으로 재생됩니다.

```csv
# filename,expression,gesture
my_meme.gif,happy,thumbs-up
```

표정은 위의 7개 이름 또는 `unknown`, 손동작은 `none`, `fist`, `open-palm`, `peace`,
`pointing`, `thumbs-up`, `other` 중 하나를 사용합니다. 빌드할 때 전체 `assets` 계층이
실행 파일 옆에 동기화됩니다.
