import os
import sys
import itertools
import json
import tempfile
import gc
import re
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageChops
from pillow_heif import register_heif_opener
from PyQt5.QtCore import Qt, QEvent, QObject, QUrl, QPointF, QRectF, QThread, QCoreApplication, QTimer, QPropertyAnimation, QVariantAnimation, QEasingCurve, QSize, QByteArray, QMimeData, pyqtProperty, pyqtSignal, qInstallMessageHandler
from PyQt5.QtGui import QColor, QCursor, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPixmapCache
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import (QApplication, QAction, QButtonGroup, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QShortcut, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar, QMenu, QDialog, QLineEdit, QKeySequenceEdit, QTabWidget, QTextEdit, QPlainTextEdit, QListWidget, QListWidgetItem, QListView, QCheckBox, QToolButton, QStackedWidget, QScrollArea, QSplitter, QSizePolicy, QGraphicsOpacityEffect, QAbstractButton, QProxyStyle, QStyle, QVBoxLayout, QWidget, QTabBar)

# Qt's bundled plugin path can lose non-ASCII characters on Windows.
# Set it explicitly using Python's Unicode-aware path before QApplication starts.
import PyQt5
_qt_plugins = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
if os.path.isdir(_qt_plugins):
    QCoreApplication.setLibraryPaths([_qt_plugins])
register_heif_opener()
_content_revisions = itertools.count(1)
R = Image.Resampling



SHORTCUT_DEFAULTS = {'new_document':'Ctrl+N', 'open':'Ctrl+O', 'save':'Ctrl+S',
                     'undo':'Ctrl+Z', 'redo':'Ctrl+Y', 'copy':'Ctrl+C',
                     'paste':'Ctrl+V', 'transform':'Ctrl+T', 'move':'V', 'rectangle':'M', 'lasso':'L'}
SHORTCUT_LABELS = {'new_document':'새 문서', 'open':'사진 불러오기', 'save':'저장',
                   'undo':'실행 취소', 'redo':'다시 실행', 'copy':'복사',
                   'paste':'붙여넣기', 'transform':'레이어 변형', 'move':'이동 도구', 'rectangle':'사각형 선택', 'lasso':'올가미 선택'}
FIXED_SHORTCUTS = {'Ctrl+Shift+N','Ctrl+Shift+S','Ctrl+W','Ctrl+Q','Ctrl+D','Ctrl+A','Ctrl+Shift+D','Ctrl+Shift+I','Ctrl+Alt+Shift+W','Ctrl+Alt+Shift+S','Shift+F6','Ctrl+Shift+Z','Del','Return','Enter','Esc'}
SETTINGS_PATH = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'MiniPhotoshop', 'settings.json')
_RUNTIME_MEMORY_SAVING = True


def default_settings():
    return {'theme':'dark','language':'ko','history_mb':512,'memory_saving':True,'shortcuts':dict(SHORTCUT_DEFAULTS),'export':{'format':'PNG','quality':90,'transparency':True,'animate':True,'frame_ms':2000,'repeat':True}}


def load_settings():
    config=default_settings()
    try:
        with open(SETTINGS_PATH,'r',encoding='utf-8') as f: saved=json.load(f)
        if saved.get('theme') in ('dark','light'): config['theme']=saved['theme']
        if saved.get('language') in ('ko','en'): config['language']=saved['language']
        export=saved.get('export',{})
        if isinstance(export,dict):
            if export.get('format') in ('PNG','JPEG','GIF'):config['export']['format']=export['format']
            if isinstance(export.get('quality'),int):config['export']['quality']=max(1,min(100,export['quality']))
            if isinstance(export.get('transparency'),bool):config['export']['transparency']=export['transparency']
            if isinstance(export.get('animate'),bool):config['export']['animate']=export['animate']
            if isinstance(export.get('frame_ms'),int):config['export']['frame_ms']=max(20,min(60000,export['frame_ms']))
            if isinstance(export.get('repeat'),bool):config['export']['repeat']=export['repeat']
        limit=int(saved.get('history_mb',512));config['history_mb']=max(64,min(8192,limit))
        if isinstance(saved.get('memory_saving'),bool):config['memory_saving']=saved['memory_saving']
        keys=saved.get('shortcuts',{})
        if isinstance(keys,dict):
            proposed={name:keys.get(name,default) for name,default in SHORTCUT_DEFAULTS.items()}
            values=[QKeySequence(str(value)).toString(QKeySequence.PortableText) for value in proposed.values()]
            reserved={QKeySequence(item).toString(QKeySequence.PortableText) for item in FIXED_SHORTCUTS}
            if all(values) and len(set(values))==len(values) and not set(values)&reserved and all(QKeySequence(v).count()==1 for v in values):
                config['shortcuts']=dict(zip(SHORTCUT_DEFAULTS,values))
    except (OSError,ValueError,TypeError,AttributeError): pass
    return config


def save_settings(config):
    directory=os.path.dirname(SETTINGS_PATH);os.makedirs(directory,exist_ok=True)
    fd,path=tempfile.mkstemp(prefix='.settings-',suffix='.json',dir=directory)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(config,f,ensure_ascii=False,indent=2)
        os.replace(path,SETTINGS_PATH)
    finally:
        if os.path.exists(path):os.unlink(path)


def history_memory_bytes(states):
    seen=set();total=0
    for state in states:
        images=list(state[:5])
        if len(state)>6:images.extend(layer.get('image') for layer in state[6])
        for image in images:
            if not isinstance(image,Image.Image) or id(image) in seen:continue
            seen.add(id(image));total+=image.width*image.height*len(image.getbands())
    return total


def trim_history(history, future, budget):
    while (history or future) and history_memory_bytes(history+future)>budget:
        if history: history.pop(0)
        else: future.pop(0)


def make_inference_session(path):
    import onnxruntime as ort
    options=ort.SessionOptions()
    if _RUNTIME_MEMORY_SAVING:
        options.enable_cpu_mem_arena=False
        options.enable_mem_pattern=False
    return ort.InferenceSession(path,sess_options=options,providers=['CPUExecutionProvider'])


LIGHT_COLORS = {
 '#151922':'#f2f5fa','#1d2330':'#f6f8fc','#303030':'#e8edf5','#191f29':'#e6ebf3',
 '#1b202a':'#e7edf5','#202630':'#e6ebf2','#20242e':'#dce3ed','#252e3b':'#ffffff',
 '#252c38':'#f3f6fa','#252d3a':'#ffffff','#252d3b':'#ffffff','#2a3342':'#ffffff',
 '#2b3545':'#ffffff','#2c3544':'#ffffff','#293442':'#ffffff','#303949':'#edf1f7',
 '#303c4e':'#dce6f4','#354156':'#dce6f5','#354155':'#e4ebf4','#3b485d':'#dbe5f2',
 '#3b3d42':'#edf1f7','#3a4659':'#d3dce9','#333943':'#d9e0e9','#44536b':'#cbd8e9',
 '#44536a':'#cbd8e9','#40516b':'#d1def0','#42536e':'#d1def0','#526482':'#cbdaf0',
 '#345946':'#cce9dc','#61383f':'#f6d6d8','#3a4657':'#b7c3d2','#3b485a':'#b7c3d2',
 '#303849':'#c5cfdb','#303a49':'#cbd5e2','#465266':'#b7c3d2','#394556':'#cbd4e0','#414d60':'#b8c5d6',
 '#536075':'#b2c0d0','#566780':'#b2c0d0','#566479':'#b2c0d0','#505866':'#b2c0d0',
 '#72757b':'#a7b5c8','#52647d':'#b2c0d0','#38414e':'#c8d0da',
 '#f4f7fc':'#253246','#f5f7fb':'#253246','#f3f6fc':'#253246','#eef2fa':'#253246',
 '#eef3fa':'#253246','#edf2fa':'#253246','#e9edf5':'#253246','#e8edf5':'#253246',
 '#e7edf7':'#253246','#e7eaf0':'#253246','#e6edf7':'#253246','#eff4fc':'#253246',
 '#d1dae8':'#46566c','#c2ccdb':'#53667d','#b5c2d4':'#53667d','#aebdd1':'#53667d',
 '#b9c6d7':'#53667d','#aab7c9':'#53667d','#c8d5e7':'#53667d','#ffffff':'#253246',
 '#707b8b':'#8e9aaa','#778391':'#8e9aaa','#7b8798':'#8e9aaa','#637080':'#8e9aaa'
}

def themed_stylesheet(style, theme):
    if theme != 'light': return style
    return re.sub(r'#[0-9a-fA-F]{6}',lambda m:LIGHT_COLORS.get(m.group(0).lower(),m.group(0)),style).replace('color: white','color: #253246').replace('color:white','color:#253246')


ENGLISH = {
 '미니 포토샵':'Mini Photoshop','설정':'Settings','파일':'File','이미지':'Image','선택':'Select','모두':'All','다시 선택':'Reselect','반전':'Inverse','수정':'Modify',
 '새로 만들기...':'New...', '내보내기':'Export', 'PNG로 빠른 내보내기...':'Quick Export as PNG...', '내보내기 형식...':'Export As...', '내보내기 기본 설정...':'Export Preferences...', '웹용으로 저장...':'Save for Web...',
 '테두리...':'Border...', '매끄럽게...':'Smooth...', '확대...':'Expand...', '축소...':'Contract...', '페더...':'Feather...',
 '반경 (원본 픽셀)':'Radius (original pixels)', '선택 영역 테두리':'Border Selection', '선택 영역 매끄럽게':'Smooth Selection', '선택 영역 확대':'Expand Selection', '선택 영역 축소':'Contract Selection', '선택 영역 페더':'Feather Selection',
 '레이어별 애니메이션':'Animate visible layers', '프레임 간격':'Frame duration', '반복 재생':'Loop animation',
 '아래쪽 레이어부터 순서대로 하나씩 쌓입니다.':'Visible layers accumulate from bottom to top.',
 '애니메이션 GIF를 만들려면 보이는 레이어가 두 개 이상 필요합니다.':'At least two visible layers are required for animation.',
 'GIF 애니메이션을 만들고 있습니다':'Creating animated GIF',
 '파일 설정':'File settings','형식':'Format','투명 배경 유지':'Preserve transparency','JPG 품질':'JPG quality',
 'JPG는 투명 배경이 흰색으로 저장됩니다. GIF는 투명·불투명만 지원합니다.':'JPG uses a white background. GIF supports binary transparency only.', '내보내기 형식':'Export As', '내보내기 기본 설정':'Export Preferences',
 '새 문서...':'New document...', '새 탭':'New tab','사진 불러오기...':'Open image...',
 '사진 붙여넣기':'Paste image','저장':'Save','다른 이름으로 저장...':'Save as...',
 '투명 PNG 파일 복사':'Copy transparent PNG file', '선택 레이어 PNG 저장...':'Save selected layer as PNG...', 'PPT로 내보내기...':'Export PowerPoint...', '설정...':'Settings...', '탭 닫기':'Close tab','종료':'Exit',
 '이미지 크기…':'Image size…','캔버스 크기…':'Canvas size…','이미지 크기':'Image size','캔버스 크기':'Canvas size',
 '선택 반전':'Invert selection','인물 자동 허용치':'Person selection tolerance','선택 크기':'Selection size',
 '내비게이터':'Navigator','인식 품질':'Detection quality','레이어':'Layers','정밀 (느림)':'Precise (slower)','빠름 (저사양)':'Fast (low memory)',
 '캐시 · 메모리':'Cache & memory','화면':'Appearance','언어':'Language','단축키':'Shortcuts',
 '문서별 실행 취소 기록 메모리 예산':'Undo history memory budget per document',
 '기록의 예상 메모리를 문서별로 제한합니다. 값을 줄이면 오래된 실행 취소 기록이 삭제될 수 있습니다. 현재 이미지와 AI 모델을 포함한 프로그램 전체 RAM 한도는 아닙니다.':'Limits estimated undo history memory per document. Reducing this can remove older undo steps. This does not limit total application RAM, images, or AI models.',
 'AI 인식 메모리 절약 사용':'Reduce AI memory usage','AI 메모리 설정은 다음 인물 자동 선택부터 적용됩니다.':'AI memory settings apply to the next person selection.',
 '임시 캐시 비우기':'Clear temporary cache','사진과 실행 취소 기록은 유지됩니다.':'Images and undo history are kept.',
 '임시 캐시를 비웠습니다. 사진과 실행 취소 기록은 유지됩니다.':'Temporary cache cleared. Images and undo history are kept.',
 '화면 테마':'Theme','다크 모드':'Dark mode','라이트 모드':'Light mode','프로그램 언어':'Application language',
 '언어 변경 후 저장을 누르세요.':'Click Save to apply the language.',
 '변경할 항목을 클릭한 후 원하는 키 조합을 누르세요.':'Click a field and press the desired key combination.',
 '단축키 기본값 복원':'Restore default shortcuts','취소':'Cancel','적용':'Apply','만들기':'Create',
 '새 문서':'New document','사진 불러오기':'Open image','실행 취소':'Undo','다시 실행':'Redo','복사':'Copy','붙여넣기':'Paste','레이어 변형':'Transform layer','이동 도구':'Move tool','사각형 선택':'Rectangular selection','올가미 선택':'Lasso selection',
 '이름':'Name','프리셋':'Preset','너비':'Width','높이':'Height','해상도':'Resolution','방향':'Orientation','배경':'Background',
 '직접 입력':'Custom','기본 방향':'Default','가로':'Landscape','세로':'Portrait','투명':'Transparent','흰색':'White',
 '기존 내용은 중앙에 유지됩니다.':'Existing content remains centered.',
 '캔버스와 모든 레이어의 크기를 함께 변경합니다. 개별 레이어는 Ctrl+T로 변형하세요.':'Resizes the canvas and every layer. Use Ctrl+T for an individual layer.',
 '맞춤':'Fit', '처리 중입니다. 잠시만 기다려 주세요.':'Processing. Please wait.',
 '이미지를 붙여넣고 있습니다':'Pasting image', '사진을 분석하고 있습니다':'Analyzing image',
 '선택 영역을 분석하고 있습니다':'Analyzing selection', '선택한 영역을 분석하고 있습니다':'Analyzing selected area',
 '사물을 찾고 있습니다':'Finding objects', '인물을 찾고 있습니다':'Finding people',
 '선택한 피사체를 찾고 있습니다':'Finding subject', 'PPT를 만들고 있습니다':'Creating PowerPoint',
 'PPT 와이드 16:9 · 1920 × 1080':'PPT Widescreen 16:9 · 1920 × 1080',
 'PPT 표준 4:3 · 1200 × 900':'PPT Standard 4:3 · 1200 × 900',
 '가로세로 비율 유지':'Maintain aspect ratio','비율 유지':'Maintain proportions',
 '선택 부분 제거':'Delete selected area','누끼따기':'Remove background','선택 해제':'Deselect',
 '피사체 선택':'Select subject','사람':'People','사물':'Objects','사진 없음':'No image','크기 없음':'No size',
}

ENGLISH.update({
 '사진을 새 레이어로 붙여넣었습니다. Ctrl+T로 크기와 위치를 조절하세요.':'Pasted as a new layer. Press Ctrl+T to resize or move it.',
 '이미지를 붙여넣고 있습니다…':'Pasting image…', '이미지를 복사했습니다. Ctrl+V로 붙여넣으세요.':'Image copied. Press Ctrl+V to paste.',
 'PNG 파일을 복사했습니다. 다른 앱에서 Ctrl+V로 붙여넣으세요.':'PNG file copied. Press Ctrl+V in another app to paste.',
 '투명 PNG로 복사했습니다. 카카오톡에서 Ctrl+V로 붙여넣으세요.':'Transparent PNG copied. Press Ctrl+V to paste.',
 '파일 메뉴에서 사진을 불러오거나 Ctrl+V로 붙여넣으세요.':'Open an image from the File menu or press Ctrl+V to paste.',
 '사진을 열어 시작하세요.':'Open an image to get started.', '선택 영역을 반전했습니다.':'Selection inverted.',
 '변형을 취소했습니다.':'Transform canceled.', '레이어 변형을 적용했습니다. Ctrl+Z로 되돌릴 수 있습니다.':'Layer transform applied. Press Ctrl+Z to undo.',
 '캔버스 크기를 변경했습니다.':'Canvas resized.', '레이어 크기를 변경했습니다. Ctrl+Z로 되돌릴 수 있습니다.':'Layer resized. Press Ctrl+Z to undo.',
 '레이어를 이동했습니다. Ctrl+Z로 되돌릴 수 있습니다.':'Layer moved. Press Ctrl+Z to undo.',
 '사각형 선택: 드래그 · Shift 추가 · Alt 빼기 · Delete 삭제':'Rectangular selection: drag · Shift to add · Alt to subtract · Delete to remove',
 '올가미: 마우스로 원하는 영역의 경계를 그리세요. Shift 추가 · Alt 빼기':'Lasso: draw around the area. Shift to add · Alt to subtract.',
 '선택한 레이어를 표시한 뒤 작업해 주세요.':'Make the selected layer visible before editing.',
 '선택한 레이어 위에 사각형을 그려 주세요.':'Draw a rectangle over the selected layer.',
 '선택한 레이어의 배경을 제거했습니다.':'Background removed from the selected layer.',
 '선택한 레이어에서 선택 영역을 제거했습니다.':'Selected area removed from the selected layer.',
 '선택을 해제했습니다.':'Selection cleared.', '이전 작업으로 되돌렸습니다.':'Undone.', '되돌린 작업을 다시 실행했습니다.':'Redone.',
 '숨겨진 레이어는 복사할 수 없습니다. 레이어를 표시해 주세요.':'Hidden layers cannot be copied. Make the layer visible first.',
 '선택된 영역이 없습니다.':'There is no selected area.', '선택한 레이어에 복사할 내용이 없습니다.':'There is nothing to copy on the selected layer.',
 '선택한 영역에 보이는 픽셀이 없습니다.':'There are no visible pixels in the selected area.',
 '처리하지 못했습니다.':'The operation could not be completed.', '설정을 저장했습니다.':'Settings saved.',
 '선택한 레이어에서 대상을 찾지 못했습니다. 범위를 넓히거나 선택 해제 후 다시 시도하세요.':'No subject found. Expand or clear the selection and try again.',
 '피사체의 점선 윤곽을 확인한 뒤 누끼따기를 누르세요.':'Check the selection outline, then select Remove background.',
 '피사체 선택':'Select subject', '인물 자동 선택':'Select people', '사물 자동 선택':'Select objects',
 '빠른 선택':'Quick selection', '자동 선택':'Auto selection', '객체 선택':'Object selection', '삭제':'Delete',
 '이동 도구: 드래그로 이동 · Shift를 누르면 가로 또는 세로로 이동 · 조절점으로 크기 변경':'Move tool: drag to move · hold Shift to constrain direction · drag handles to resize.',
 '선택 추가: 빠진 부분을 올가미로 둘러싸면 기존 선택에 추가됩니다.':'Add to selection: draw around the missing area.',
 '선택 빼기: 제외할 부분을 올가미로 둘러싸세요.':'Subtract from selection: draw around the area to remove.',
 '객체 선택: 현재 레이어에서 사람이나 사물을 사각형으로 감싸세요. Shift 추가 · Alt 빼기':'Object selection: draw around a person or object on the selected layer. Shift adds · Alt subtracts.',
 '자동 선택: 선택할 피사체를 클릭하세요. Shift 추가 · Alt 빼기':'Auto selection: click a subject. Shift adds · Alt subtracts.',
 '빠른 선택: 비슷한 색 영역을 드래그하세요. Alt를 누르면 선택에서 제외합니다.':'Quick selection: drag over similar colors. Hold Alt to subtract.',
 '선택한 레이어의 영역에서 피사체를 찾고 있습니다…':'Finding a subject in the selected layer…',
 '클릭한 피사체를 찾고 있습니다…':'Finding the clicked subject…',
 '클릭한 대상을 찾지 못했습니다. 빠른 선택 또는 올가미 도구를 사용해 주세요.':'No subject found. Try Quick Selection or the Lasso tool.',
 '클릭한 피사체를 선택했습니다.':'Selected the clicked subject.',
 '영역에서 피사체를 찾지 못했습니다. 대상을 여유 있게 감싸거나 빠른 선택을 사용해 주세요.':'No subject found. Draw a larger rectangle or use Quick Selection.',
 '객체를 선택했습니다. Shift로 추가, Alt로 제외할 수 있습니다.':'Object selected. Shift adds; Alt subtracts.',
 '선택한 레이어에서 사물을 찾고 있습니다…':'Finding objects in the selected layer…',
 '선택 범위에서 사람과 사물을 찾고 있습니다…':'Finding people and objects in the selected area…',
 '선택한 레이어 위에 선택 영역을 만들어 주세요.':'Create a selection over the selected layer.',
 '선택 범위에서 사람을 찾고 있습니다…':'Finding people in the selected area…',
 '사람을 찾고 가장자리를 세밀하게 다듬고 있습니다…':'Finding people and refining edges…',
 '선택한 피사체의 경계를 계산하고 있습니다…':'Calculating the subject boundary…',
 '펜 표시를 지웠습니다.':'Brush marks cleared.', '원본을 표시합니다. 펜 표시는 유지됩니다.':'Original displayed. Brush marks are retained.',
 '선택한 탭을 PPT 슬라이드로 내보내고 있습니다…':'Exporting selected tabs to PowerPoint…',
})

def english_status(text):
    if text in ENGLISH:return ENGLISH[text]
    result=text.replace(' 픽셀',' px')
    patterns=[(r'^레이어 선택: (.*)$',r'Layer selected: \1'),
              (r'^저장 완료: (.*)$',r'Saved: \1'),
              (r'^선택 레이어 PNG 저장 완료: (.*)$',r'Selected layer saved as PNG: \1'),
              (r'^PPT 내보내기 완료: (.*)$',r'PowerPoint exported: \1'),
              (r'^내보내기 완료: (.*)$',r'Exported: \1'),
              (r'^(.+) 레이어를 삭제했습니다\. Ctrl\+Z로 되돌릴 수 있습니다\.$',r'Layer \1 deleted. Press Ctrl+Z to undo.'),
              (r'^탭 이름을 (.+)\(으\)로 변경했습니다\.$',r'Tab renamed to \1.')]
    for pattern,replacement in patterns:
        if re.match(pattern,result):return re.sub(pattern,replacement,result)
    return result

class StatusLabel(QLabel):
    def __init__(self,text):
        self._translated_status=True;self._source_text=text
        super().__init__(text)
    def setText(self,text):
        self._source_text=text
        parent=self.window();prefs=getattr(parent,'preferences',{})
        super().setText(english_status(text) if prefs.get('language')=='en' else text)
    def refresh_language(self):self.setText(self._source_text)


def translate_widgets(root, language):
    english = language == 'en'
    for widget in [root] + root.findChildren(QWidget):
        if isinstance(widget, (QLabel, QPushButton, QCheckBox)) and not getattr(widget,'_translated_status',False):
            current=widget.text(); original=widget.property('_ko_text')
            if original is None or (current != original and current != ENGLISH.get(original,original)):
                original=current; widget.setProperty('_ko_text',original)
            translated=ENGLISH.get(original,original) if english else original
            if translated != current: widget.setText(translated)
        if isinstance(widget,QComboBox):
            for i in range(widget.count()):
                current=widget.itemText(i); original=widget.itemData(i,Qt.UserRole+21)
                if original is None: original=current;widget.setItemData(i,original,Qt.UserRole+21)
                widget.setItemText(i,ENGLISH.get(original,original) if english else original)
        if isinstance(widget,QTabWidget):
            for i in range(widget.count()):
                page=widget.widget(i); original=page.property('_ko_tab')
                if original is None: original=widget.tabText(i);page.setProperty('_ko_tab',original)
                widget.setTabText(i,ENGLISH.get(original,original) if english else original)
    for action in root.findChildren(QAction):
        original=action.property('_ko_text')
        if original is None: original=action.text();action.setProperty('_ko_text',original)
        action.setText(ENGLISH.get(original,original) if english else original)
    original=root.property('_ko_title')
    if original is None: original=root.windowTitle();root.setProperty('_ko_title',original)
    if original: root.setWindowTitle(ENGLISH.get(original,original) if english else original)


def set_native_titlebar_theme(widget, dark):
    """Use the Windows-managed title bar with matching light/dark colors."""
    if sys.platform != 'win32' or QApplication.platformName() != 'windows': return False
    try:
        import ctypes
        from ctypes import wintypes
        dwm=ctypes.windll.dwmapi
        dwm.DwmSetWindowAttribute.argtypes=[wintypes.HWND,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD]
        dwm.DwmSetWindowAttribute.restype=ctypes.c_long
        hwnd=int(widget.winId());value=ctypes.c_int(1 if dark else 0)
        for attribute in (20,19):
            if dwm.DwmSetWindowAttribute(hwnd,attribute,ctypes.byref(value),ctypes.sizeof(value))==0:return True
    except (OSError,AttributeError):pass
    return False


def qimage(image):
    rgba = np.ascontiguousarray(np.asarray(image.convert('RGBA')))
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()


def selected_mask(image, marks):
    ys, xs = np.where(marks == 1)
    if not len(xs):
        raise ValueError('남길 피사체를 초록색 펜으로 먼저 칠해 주세요.')
    h, w = marks.shape
    margin = max(15, int(max(w, h) * .08))
    x0, x1 = max(0, xs.min()-margin), min(w, xs.max()+margin+1)
    y0, y1 = max(0, ys.min()-margin), min(h, ys.max()+margin+1)
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[y0:y1, x0:x1] = cv2.GC_PR_FGD
    gc[0, :] = gc[-1, :] = cv2.GC_BGD
    gc[:, 0] = gc[:, -1] = cv2.GC_BGD
    gc[marks == 1] = cv2.GC_FGD
    gc[marks == 2] = cv2.GC_BGD
    cv2.grabCut(np.asarray(image.convert('RGB')).copy(), gc, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
    return Image.fromarray(np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype('uint8'))


def composite_layers(layers, size, exclude=None):
    base = Image.new('RGBA', size, (0,0,0,0))
    for index, layer in enumerate(layers):
        if index == exclude or not layer['visible']: continue
        image, x, y = layer['image'], layer['x'], layer['y']
        left,top = max(0,-x),max(0,-y)
        right,bottom = min(image.width,size[0]-x),min(image.height,size[1]-y)
        if right > left and bottom > top:
            base.alpha_composite(image.crop((left,top,right,bottom)), (x+left,y+top))
    return base


def preview_pixmap(image):
    w,h=image.size;scale=min(1.0,1536/max(w,h))
    thumb=image if scale==1.0 else image.resize((max(1,round(w*scale)),max(1,round(h*scale))),R.LANCZOS,reducing_gap=2.0)
    return QPixmap.fromImage(qimage(thumb))



def composite_preview(layers, canvas_size, maximum=1536, cache=None):
    cw,ch=canvas_size
    scale=min(1.0, maximum/max(cw,ch)); size=(max(1,round(cw*scale)),max(1,round(ch*scale)))
    reduced=[]
    for layer in layers:
        if not layer['visible']: continue
        image=layer['image']; x,y=layer['x'],layer['y']
        left,top=max(0,-x),max(0,-y);right,bottom=min(image.width,cw-x),min(image.height,ch-y)
        if right<=left or bottom<=top: continue
        x0,y0=round((x+left)*size[0]/cw),round((y+top)*size[1]/ch)
        x1,y1=round((x+right)*size[0]/cw),round((y+bottom)*size[1]/ch)
        if x1<=x0 or y1<=y0: continue
        key=(id(image),left,top,right,bottom,x1-x0,y1-y0)
        cached=cache.get(key) if cache is not None else None
        if cached is not None and cached[0] is image: crop=cached[1]
        else:
            source=image if (left,top,right,bottom)==(0,0,image.width,image.height) else image.crop((left,top,right,bottom))
            crop=source.resize((x1-x0,y1-y0),R.BILINEAR,reducing_gap=2.0)
            if cache is not None:
                if len(cache)>=12:cache.pop(next(iter(cache)))
                cache[key]=(image,crop)
        reduced.append({'image':crop,'x':x0,'y':y0,'visible':True})
    return QPixmap.fromImage(qimage(composite_layers(reduced,size)))


def clear_hidden_rgb(image):
    rgba=image.convert('RGBA')
    if rgba.getchannel('A').getextrema()[0] == 255:return rgba
    pixels=np.array(rgba,copy=True)
    pixels[pixels[:,:,3]==0,:3]=255
    return Image.fromarray(pixels,'RGBA')


def copy_png_file_to_clipboard(image):
    """Expose an alpha-preserving PNG as a normal copied file (CF_HDROP)."""
    directory=os.path.join(tempfile.gettempdir(),'MiniPhotoshopClipboard')
    os.makedirs(directory,exist_ok=True)
    fd,path=tempfile.mkstemp(prefix='cutout-',suffix='.png',dir=directory)
    try:
        with os.fdopen(fd,'wb') as output:clear_hidden_rgb(image).save(output,format='PNG')
        mime=QMimeData();mime.setUrls([QUrl.fromLocalFile(path)])
        if sys.platform=='win32':
            import struct
            mime.setData('application/x-qt-windows-mime;value="Preferred DropEffect"',QByteArray(struct.pack('<I',1)))
        QApplication.clipboard().setMimeData(mime)
        return path
    except Exception:
        try:os.unlink(path)
        except OSError:pass
        raise


def copy_rgba_to_clipboard(image):
    """Place straight-alpha PNG and Windows DIBV5 on the clipboard."""
    import io
    # Keep alpha and visible colors intact while clearing source RGB hidden
    # in completely transparent pixels, which some messengers display.
    rgba_image=clear_hidden_rgb(image)
    has_transparency=rgba_image.getchannel('A').getextrema()[0]<255
    data=io.BytesIO();rgba_image.save(data,format='PNG');png=data.getvalue()
    if sys.platform=='win32':
        try:
            import ctypes,struct
            from ctypes import wintypes
            user32=ctypes.windll.user32;kernel=ctypes.windll.kernel32
            kernel.GlobalAlloc.argtypes=[wintypes.UINT,ctypes.c_size_t];kernel.GlobalAlloc.restype=wintypes.HGLOBAL
            kernel.GlobalLock.argtypes=[wintypes.HGLOBAL];kernel.GlobalLock.restype=ctypes.c_void_p
            kernel.GlobalUnlock.argtypes=[wintypes.HGLOBAL];kernel.GlobalFree.argtypes=[wintypes.HGLOBAL]
            user32.SetClipboardData.argtypes=[wintypes.UINT,wintypes.HANDLE];user32.SetClipboardData.restype=wintypes.HANDLE
            user32.RegisterClipboardFormatW.argtypes=[wintypes.LPCWSTR];user32.RegisterClipboardFormatW.restype=wintypes.UINT
            if has_transparency:
                # A file drop lets messengers that flatten clipboard bitmaps
                # send the original alpha-preserving PNG as a file instead.
                directory=os.path.join(tempfile.gettempdir(),'MiniPhotoshopClipboard')
                os.makedirs(directory,exist_ok=True)
                fd,path=tempfile.mkstemp(prefix='cutout-',suffix='.png',dir=directory)
                with os.fdopen(fd,'wb') as output:output.write(png)
            rgba=np.asarray(rgba_image);height,width=rgba.shape[:2]
            pixels=np.ascontiguousarray(rgba[::-1,:,[2,1,0,3]]).tobytes()
            header=bytearray(124)
            struct.pack_into('<IiiHHIIiiII',header,0,124,width,height,1,32,3,len(pixels),0,0,0,0)
            struct.pack_into('<IIIII',header,40,0x00ff0000,0x0000ff00,0x000000ff,0xff000000,0x73524742)
            def put(fmt, payload):
                handle=kernel.GlobalAlloc(0x0002,len(payload))
                if not handle:raise MemoryError('클립보드 메모리 부족')
                ptr=kernel.GlobalLock(handle)
                if not ptr:kernel.GlobalFree(handle);raise OSError('클립보드를 잠글 수 없습니다')
                ctypes.memmove(ptr,payload,len(payload));kernel.GlobalUnlock(handle)
                if not user32.SetClipboardData(fmt,handle):kernel.GlobalFree(handle);raise OSError('클립보드 형식을 설정할 수 없습니다')
            if not user32.OpenClipboard(None): raise OSError('클립보드를 사용할 수 없습니다')
            try:
                if not user32.EmptyClipboard(): raise OSError('클립보드를 비울 수 없습니다')
                if has_transparency:
                    # Retain the PNG file and true-alpha formats. Some Windows
                    # messengers only accept CF_DIB, so offer a white-composited
                    # compatibility image instead of exposing discarded RGB.
                    drop=struct.pack('<IiiII',20,0,0,0,1)+(path+'\0\0').encode('utf-16-le')
                    put(15,drop)
                    put(user32.RegisterClipboardFormatW('PNG'),png)
                    put(17,bytes(header)+pixels)
                    background=Image.new('RGBA',(width,height),'white')
                    background.alpha_composite(rgba_image)
                    flattened=np.asarray(background)
                    dib_pixels=np.ascontiguousarray(flattened[::-1,:,[2,1,0,3]]).tobytes()
                else:
                    put(user32.RegisterClipboardFormatW('PNG'),png)
                    put(17,bytes(header)+pixels)
                    dib_pixels=pixels
                dib_header=struct.pack('<IiiHHIIiiII',40,width,height,1,32,0,len(dib_pixels),0,0,0,0)
                put(8,dib_header+dib_pixels)
            finally:user32.CloseClipboard()
            return
        except (OSError,MemoryError,ValueError):pass
    mime=QMimeData();mime.setImageData(qimage(image));mime.setData('image/png',QByteArray(png))
    QApplication.clipboard().setMimeData(mime)


def fill_small_mask_holes(mask):
    binary = (np.asarray(mask) >= 128).astype(np.uint8)
    h,w = binary.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1-binary, 8)
    limit = max(4, min(180, round(h*w*0.00025)))
    for i in range(1,count):
        x,y,bw,bh,area = stats[i]
        if area <= limit and x > 0 and y > 0 and x+bw < w and y+bh < h:
            binary[labels == i] = 1
    return binary*255


def soften_mask(mask):
    data = np.asarray(mask.convert('L'), dtype=np.uint8)
    # Smooth the visible selection contour first, then keep a narrow alpha
    # transition. This avoids both jagged hard edges and a broad background halo.
    selected = (data >= 128).astype(np.float32)
    smooth = cv2.GaussianBlur(selected, (0,0), 1.15)
    # Preserve very thin selected details instead of erasing them altogether.
    smooth = np.where((selected > 0) & (smooth < 0.35), 0.35, smooth)
    alpha = np.clip((smooth - 0.15) / 0.70, 0, 1)
    return Image.fromarray((alpha * 255).astype(np.uint8))


def prepare_selection_mask(selection, size):
    """Keep explicitly softened selections; only smooth hard-edged masks."""
    original=np.asarray(selection.convert('L'),dtype=np.uint8)
    resized=selection.resize(size,R.LANCZOS)
    if np.any((original>0)&(original<255)):
        return resized
    return soften_mask(resized)


def finish_person_mask(image, mask):
    """Smooth the final combined mask against photo edges, preserving its interior."""
    alpha = np.asarray(mask.convert('L'), dtype=np.uint8)
    if not np.any(alpha): return mask
    rgb = np.asarray(image.convert('RGB'))
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    values = alpha.astype(np.float32) / 255.0
    size = (9,9)
    mean_i = cv2.boxFilter(guide, -1, size)
    mean_p = cv2.boxFilter(values, -1, size)
    var_i = cv2.boxFilter(guide*guide, -1, size) - mean_i*mean_i
    covariance = cv2.boxFilter(guide*values, -1, size) - mean_i*mean_p
    a = covariance / (var_i + 0.002)
    b = mean_p - a*mean_i
    filtered = cv2.boxFilter(a, -1, size)*guide + cv2.boxFilter(b, -1, size)
    out = (np.clip(filtered,0,1)*255).astype(np.uint8)
    out[(alpha == 255) & (out >= 250)] = 255
    out[(alpha == 0) & (out <= 5)] = 0
    # Recheck after combining model masks; combining can leave tiny pinholes.
    filled = fill_small_mask_holes(out)
    out[(filled > 0) & (out < 128)] = 255
    return Image.fromarray(out)


def refine_person_edges(image, mask):
    """Refine the coarse model boundary against the actual image pixels."""
    binary = fill_small_mask_holes(mask)
    if not np.any(binary) or np.all(binary): return Image.fromarray(binary)
    h, w = binary.shape
    # Work near the model boundary; confidently selected pixels remain foreground.
    radius = max(2, min(12, round(max(w, h) * 0.004)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    core = cv2.erode(binary, kernel) > 0
    border = cv2.dilate(binary, kernel) > 0
    if not np.any(core) or np.all(border): return Image.fromarray(binary)
    labels = np.full((h,w), cv2.GC_BGD, dtype=np.uint8)
    labels[border] = cv2.GC_PR_BGD
    labels[binary > 127] = cv2.GC_PR_FGD
    labels[core] = cv2.GC_FGD
    try:
        rgb = np.asarray(image.convert('RGB')).copy()
        cv2.grabCut(rgb, labels, None, np.zeros((1,65),np.float64),
                    np.zeros((1,65),np.float64), 2, cv2.GC_INIT_WITH_MASK)
        refined = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        # A subpixel transition reduces stair-steps without blurring the photo itself.
        refined = fill_small_mask_holes(refined)
        refined = cv2.GaussianBlur(refined, (3,3), 0.55)
        return Image.fromarray(refined)
    except cv2.error:
        return Image.fromarray(binary)


def refine_person_matte(image, mask):
    """Add soft portrait edges near the detected person, keeping the mask core."""
    import onnxruntime as ort
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'modnet.onnx')
    if not os.path.exists(model_path):
        return mask
    base = np.asarray(mask, dtype=np.uint8)
    binary = (base >= 128).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1: return mask
    try:
        sess = make_inference_session(model_path)
        rgb = np.asarray(image.convert('RGB'))
        h, w = binary.shape
        combined = np.zeros((h,w), dtype=np.uint8)
        for i in range(1, min(count, 20)):
            x,y,bw,bh,area = stats[i]
            if area < max(25, w*h*0.0001):
                combined[labels == i] = base[labels == i]
                continue
            pad = max(8, round(max(bw,bh) * .14))
            x0,y0,x1,y1 = max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)
            crop = rgb[y0:y1,x0:x1]; ch,cw = crop.shape[:2]
            scale = min(512/max(1,min(ch,cw)), 768/max(ch,cw))
            iw = max(32, round(cw*scale/32)*32); ih = max(32, round(ch*scale/32)*32)
            tensor = cv2.resize(crop,(iw,ih),interpolation=cv2.INTER_LINEAR).astype(np.float32)/127.5-1.0
            tensor = np.ascontiguousarray(tensor.transpose(2,0,1)[None])
            matte = sess.run(None, {sess.get_inputs()[0].name:tensor})[0][0,0]
            matte = cv2.resize(matte,(cw,ch),interpolation=cv2.INTER_LINEAR)
            matte = (np.clip(matte,0,1)*255).astype(np.uint8)
            local = (labels[y0:y1,x0:x1] == i).astype(np.uint8)*255
            radius = max(2, min(16, round(max(bw,bh)*0.025)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1,radius*2+1))
            core = cv2.erode(local,kernel)
            allowed = cv2.dilate(local,kernel)
            # Keep the reliable interior. Limit matting to the nearby edge so
            # other people or furniture do not become selected by the crop.
            refined = np.where(core > 0, 255, np.where(allowed > 0, matte, 0)).astype(np.uint8)
            combined[y0:y1,x0:x1] = np.maximum(combined[y0:y1,x0:x1], refined)
        return Image.fromarray(combined)
    except (ValueError, cv2.error, RuntimeError):
        return mask


def select_people(image, roi=None, focus=None, point=None, tolerance=30, quality='precise', _supplement=True):
    """Select each detected person independently with a lightweight CPU model."""
    import onnxruntime as ort
    filename = 'yolov8m-seg.onnx' if quality == 'precise' else 'yolov8s-seg.onnx'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', filename)
    if not os.path.exists(path):
        raise FileNotFoundError('선택 모델이 없습니다. models 폴더의 ' + filename + ' 파일을 확인해 주세요.')
    sess = make_inference_session(path)
    rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    input_h, input_w = sess.get_inputs()[0].shape[2:4]
    scale = min(input_w / w, input_h / h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    left, top = (input_w-rw)//2, (input_h-rh)//2
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    canvas[top:top+rh, left:left+rw] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2,0,1)[None], dtype=np.float32) / 255.0
    predictions, proto = sess.run(None, {sess.get_inputs()[0].name: tensor})
    det = predictions[0].T
    # COCO class 0 is person; require it to be the most likely class.
    if point is None:
        scores = det[:, 4]
        candidates = det[(scores > 0.22) & (np.argmax(det[:, 4:84], axis=1) == 0)]
        candidate_classes = np.zeros(len(candidates), dtype=np.int32)
    else:
        scores = det[:, 4:84].max(axis=1)
        keep = scores > 0.25
        candidates = det[keep].copy()
        candidate_classes = np.argmax(candidates[:, 4:84], axis=1)
        candidates[:, 4] = scores[keep]
    if not len(candidates): return Image.new('L', (w,h), 0)
    boxes = candidates[:, :4].copy()
    boxes[:, 0] -= boxes[:, 2]/2; boxes[:, 1] -= boxes[:, 3]/2
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), candidates[:,4].tolist(), 0.22, 0.5)
    if len(indices) == 0: return Image.new('L', (w,h), 0)
    out = np.zeros((h,w), dtype=np.uint8)
    semantic_reference = np.zeros((h,w), dtype=np.uint8)
    point_rank = None
    point_is_person = False
    prototypes = proto[0].reshape(32,-1)
    for i in np.asarray(indices).reshape(-1):
        row = candidates[i]; x,y,bw,bh = boxes[i]
        if roi is not None:
            rx0, ry0, rx1, ry1 = roi
            cx, cy = (x+bw/2-left)/scale, (y+bh/2-top)/scale
            if not (rx0 <= cx <= rx1 and ry0 <= cy <= ry1): continue
        mask = (row[84:] @ prototypes).reshape(proto.shape[2], proto.shape[3])
        mask = cv2.resize(mask, (input_w,input_h), interpolation=cv2.INTER_LINEAR)
        # Restrict each mask to its detection box before undoing letterboxing.
        x0, y0 = max(0,int(x)), max(0,int(y))
        x1, y1 = min(input_w,int(x+bw+1)), min(input_h,int(y+bh+1))
        cropped = np.zeros((input_h,input_w),dtype=np.uint8)
        # Lower logit cutoff at higher tolerance keeps more uncertain edge pixels.
        tolerance = max(0, min(100, int(tolerance)))
        cutoff = (30 - tolerance) * 0.03
        cropped[y0:y1,x0:x1] = (mask[y0:y1,x0:x1] > cutoff).astype(np.uint8)*255
        cropped = cropped[top:top+rh,left:left+rw]
        instance = cv2.resize(cropped, (w,h), interpolation=cv2.INTER_NEAREST)
        reference = np.zeros((input_h,input_w), dtype=np.uint8)
        reference[y0:y1,x0:x1] = (mask[y0:y1,x0:x1] > 0.35).astype(np.uint8)*255
        reference = cv2.resize(reference[top:top+rh,left:left+rw], (w,h), interpolation=cv2.INTER_NEAREST)
        if point is not None:
            px, py = int(point[0]), int(point[1])
            if not (0 <= px < w and 0 <= py < h and instance[py,px]): continue
        if focus is not None:
            region = np.asarray(focus.resize((w,h), R.NEAREST)) > 127
            pixels = instance > 0
            if not np.any(pixels) or np.count_nonzero(pixels & region) / np.count_nonzero(pixels) < 0.35:
                continue
        if point is not None:
            is_person = int(candidate_classes[i]) == 0
            # When several detected objects cover the clicked pixel, prefer a
            # person and choose the smallest matching instance among people.
            rank = (int(is_person), -int(np.count_nonzero(instance)))
            if point_rank is not None and rank <= point_rank: continue
            point_rank, point_is_person = rank, is_person
            out = instance.copy()
            semantic_reference = reference.copy()
        else:
            out = np.maximum(out, instance)
            semantic_reference = np.maximum(semantic_reference, reference)
    refined = refine_person_edges(image, out)
    if point is None or point_is_person:
        refined = refine_person_matte(image, refined)
        # High tolerance may soften the person's boundary, but must not grow
        # freely across an entire adjacent object such as a tablecloth.
        if np.any(semantic_reference):
            radius = max(2, min(6, round(max(w,h) * 0.004)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
            guard = cv2.dilate(semantic_reference, kernel)
            guard = cv2.GaussianBlur(guard, (3,3), 0.6)
            refined = Image.fromarray(np.minimum(np.asarray(refined, dtype=np.uint8), guard))
    if quality == 'precise' and point is None and _supplement:
        # Some occluded hands, shoes and legs are found by the smaller model
        # but missed by the larger one. Add only components that substantially
        # overlap a person already detected by the precise model.
        alternate = select_people(image, roi, focus, point, tolerance, 'fast', False)
        primary = np.asarray(refined, dtype=np.uint8)
        secondary = np.asarray(alternate, dtype=np.uint8)
        if np.any(primary >= 128):
            count, labels, stats, _ = cv2.connectedComponentsWithStats((secondary >= 128).astype(np.uint8), 8)
            result = primary.copy()
            for i in range(1, count):
                region = labels == i
                area = stats[i, cv2.CC_STAT_AREA]
                if area and np.count_nonzero(region & (primary >= 128)) / area >= 0.5:
                    result[region] = np.maximum(result[region], secondary[region])
            refined = Image.fromarray(result)
        elif np.any(secondary):
            refined = alternate
    if quality == 'precise' and point is None and np.any(np.asarray(refined)):
        refined = finish_person_mask(image, refined)
    return refined


def select_solid_background_subjects(image, focus=None):
    """Separate artwork on a uniform border without relying on object classes."""
    rgba=np.asarray(image.convert('RGBA'));rgb=rgba[:,:,:3];alpha=rgba[:,:,3]
    h,w=alpha.shape
    if min(w,h)<4:return None
    border=np.concatenate((rgb[0],rgb[-1],rgb[1:-1,0],rgb[1:-1,-1]))
    border_alpha=np.concatenate((alpha[0],alpha[-1],alpha[1:-1,0],alpha[1:-1,-1]))
    opaque_border=border[border_alpha>=250]
    if len(opaque_border)<len(border)*0.95:return None
    color=np.median(opaque_border,axis=0).astype(np.int16)
    if np.mean(np.max(np.abs(opaque_border.astype(np.int16)-color),axis=1)<=12)<0.97:return None
    similar=(np.max(np.abs(rgb.astype(np.int16)-color),axis=2)<=12) & (alpha>=250)
    count,labels=cv2.connectedComponents(similar.astype(np.uint8),8)
    edge_labels=np.unique(np.concatenate((labels[0],labels[-1],labels[1:-1,0],labels[1:-1,-1])))
    edge_labels=edge_labels[edge_labels!=0]
    background=np.isin(labels,edge_labels) if count>1 else np.zeros((h,w),dtype=bool)
    foreground=(~background)&(alpha>0)
    if not np.any(foreground) or foreground.mean()>0.90:return None
    if focus is not None:
        selected=np.asarray(focus.resize((w,h),R.NEAREST))>127
        _,components,stats,_=cv2.connectedComponentsWithStats(foreground.astype(np.uint8),8)
        kept=np.zeros((h,w),dtype=bool)
        for component in range(1,len(stats)):
            pixels=components==component
            if np.count_nonzero(pixels & selected)>=max(1,stats[component,cv2.CC_STAT_AREA]*0.35):kept|=pixels
        foreground=kept
    return Image.fromarray(np.where(foreground,alpha,0).astype(np.uint8))


def select_objects(image, focus=None, tolerance=30, quality='precise', roi=None, include_people=False):
    """Select supported non-person object instances in the active layer."""
    if roi is not None:
        from PIL import ImageDraw
        w,h=image.size;rx0,ry0,rx1,ry1=roi
        rx0=max(0,min(w-1,int(rx0)));ry0=max(0,min(h-1,int(ry0)))
        rx1=max(rx0+1,min(w,int(np.ceil(rx1))));ry1=max(ry0+1,min(h,int(np.ceil(ry1))))
        pad=max(8,round(max(rx1-rx0,ry1-ry0)*0.2))
        x0,y0=max(0,rx0-pad),max(0,ry0-pad);x1,y1=min(w,rx1+pad),min(h,ry1+pad)
        crop=image.crop((x0,y0,x1,y1));region=Image.new('L',crop.size,0)
        ImageDraw.Draw(region).rectangle((rx0-x0,ry0-y0,rx1-x0-1,ry1-y0-1),fill=255)
        result=select_objects(crop,region,tolerance,quality,include_people=include_people)
        if not result.getbbox():
            # GrabCut is a fallback for objects outside the detector's classes.
            cw,ch=crop.size;rect=(rx0-x0,ry0-y0,rx1-rx0,ry1-ry0)
            if rect[2]>=2 and rect[3]>=2 and rect[2]*rect[3]<cw*ch:
                try:
                    data=np.asarray(crop.convert('RGB')).copy();mask=np.zeros((ch,cw),np.uint8)
                    cv2.grabCut(data,mask,rect,None,None,3,cv2.GC_INIT_WITH_RECT)
                    result=Image.fromarray(np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8))
                except cv2.error: pass
        output=Image.new('L',(w,h),0);output.paste(result,(x0,y0));return output
    flat_result=select_solid_background_subjects(image,focus)
    if flat_result is not None:return flat_result
    filename = 'yolov8m-seg.onnx' if quality == 'precise' else 'yolov8s-seg.onnx'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', filename)
    if not os.path.exists(path): raise FileNotFoundError('선택 모델이 없습니다: ' + filename)
    session = make_inference_session(path)
    rgb = np.asarray(image.convert('RGB')); h,w=rgb.shape[:2]
    ih,iw=session.get_inputs()[0].shape[2:4];scale=min(iw/w,ih/h)
    rw,rh=max(1,round(w*scale)),max(1,round(h*scale));left,top=(iw-rw)//2,(ih-rh)//2
    model_image=np.full((ih,iw,3),114,dtype=np.uint8)
    model_image[top:top+rh,left:left+rw]=cv2.resize(rgb,(rw,rh),interpolation=cv2.INTER_LINEAR)
    tensor=np.ascontiguousarray(model_image.transpose(2,0,1)[None],dtype=np.float32)/255.0
    prediction,proto=session.run(None,{session.get_inputs()[0].name:tensor})
    det=prediction[0].T; classes=np.argmax(det[:,4:84],axis=1);scores=det[:,4:84].max(axis=1)
    rows=det[((classes!=0)|include_people)&(scores>0.20)].copy()
    if len(rows)==0:return Image.new('L',(w,h),0)
    rows[:,4]=rows[:,4:84].max(axis=1);boxes=rows[:,:4].copy();boxes[:,0]-=boxes[:,2]/2;boxes[:,1]-=boxes[:,3]/2
    indices=cv2.dnn.NMSBoxes(boxes.tolist(),rows[:,4].tolist(),0.20,0.5)
    output=np.zeros((h,w),dtype=np.uint8);prototypes=proto[0].reshape(32,-1)
    region=np.asarray(focus.resize((w,h),R.NEAREST))>127 if focus is not None else None
    for i in np.asarray(indices).reshape(-1):
        x,y,bw,bh=boxes[i];x0,y0=max(0,int(x)),max(0,int(y));x1,y1=min(iw,int(x+bw+1)),min(ih,int(y+bh+1))
        logits=(rows[i,84:]@prototypes).reshape(proto.shape[2],proto.shape[3]);logits=cv2.resize(logits,(iw,ih),interpolation=cv2.INTER_LINEAR)
        mask=np.zeros((ih,iw),dtype=np.uint8);mask[y0:y1,x0:x1]=(logits[y0:y1,x0:x1]>(30-max(0,min(100,int(tolerance))))*0.03).astype(np.uint8)*255
        mask=cv2.resize(mask[top:top+rh,left:left+rw],(w,h),interpolation=cv2.INTER_NEAREST)
        if region is not None:
            pixels=mask>0
            if not np.any(pixels) or np.count_nonzero(pixels & region)/np.count_nonzero(pixels)<0.35:continue
        output=np.maximum(output,mask)
    return refine_person_edges(image,output) if np.any(output) else Image.fromarray(output)


class Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    def __init__(self, image, marks, automatic, roi=None, modifiers=0, focus=None, point=None, tolerance=30, quality='precise', result_size=None, result_box=None, subjects='person'):
        super().__init__()
        self.image, self.marks, self.automatic = image, marks, automatic
        self.roi, self.modifiers, self.focus, self.point = roi, modifiers, focus, point
        self.tolerance, self.quality = tolerance, quality
        self.result_size, self.result_box = result_size, result_box
        self.subjects = subjects
    def run(self):
        try:
            if self.automatic:
                result = select_objects(self.image, self.focus, self.tolerance, self.quality, roi=self.roi, include_people=self.subjects == 'all' or (self.roi is not None and self.focus is None)) if self.subjects in ('objects','all') else select_people(self.image, self.roi, self.focus, self.point, self.tolerance, self.quality)
                if self.result_box is not None:
                    result = ImageChops.multiply(result, self.image.getchannel('A'))
                    full = Image.new('L', self.result_size, 0)
                    full.paste(result, self.result_box[:2]); result = full
            else:
                result = selected_mask(self.image, self.marks)
            self.done.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


def write_animated_gif(path, layers, canvas_size, settings, result_mask=None):
    """Export cumulative visible layers in bottom-to-top order."""
    frames=[];cw,ch=canvas_size
    accumulated=Image.new('RGBA',canvas_size,(0,0,0,0))
    mask=result_mask.resize(canvas_size,R.LANCZOS) if result_mask is not None else None
    for layer in layers:
        if not layer['visible']:continue
        image,x,y=layer['image'],layer['x'],layer['y']
        left,top=max(0,-x),max(0,-y)
        right,bottom=min(image.width,cw-x),min(image.height,ch-y)
        if right>left and bottom>top:
            accumulated.alpha_composite(image.crop((left,top,right,bottom)),(x+left,y+top))
        frame=accumulated.copy()
        if mask is not None:frame.putalpha(ImageChops.multiply(frame.getchannel('A'),mask))
        alpha=frame.getchannel('A')
        if not settings.get('transparency',True):
            background=Image.new('RGBA',(cw,ch),'white');background.alpha_composite(frame);frame=background
        indexed=frame.convert('RGB').quantize(colors=255,method=Image.Quantize.MEDIANCUT)
        if settings.get('transparency',True):
            transparent=alpha.point(lambda value:255 if value<128 else 0)
            indexed.paste(255,mask=transparent);indexed.info['transparency']=255
        frames.append(indexed)
    if len(frames)<2:raise ValueError('애니메이션 GIF에는 보이는 레이어가 두 개 이상 필요합니다.')
    duration=max(20,min(60000,round(settings.get('frame_ms',2000)/10)*10))
    options={'save_all':True,'append_images':frames[1:],'duration':duration,'disposal':2,'optimize':False}
    if settings.get('repeat',True):options['loop']=0
    if settings.get('transparency',True):options['transparency']=255
    directory=os.path.dirname(os.path.abspath(path));fd,tmp=tempfile.mkstemp(prefix='.mini-gif-',suffix='.gif',dir=directory);os.close(fd)
    try:
        frames[0].save(tmp,'GIF',**options);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


class GifExportWorker(QThread):
    done=pyqtSignal(str)
    failed=pyqtSignal(str)
    def __init__(self,path,layers,size,settings,result_mask):
        super().__init__();self.path=path;self.layers=layers;self.size=size;self.settings=settings;self.result_mask=result_mask
    def run(self):
        try:write_animated_gif(self.path,self.layers,self.size,self.settings,self.result_mask);self.done.emit(self.path)
        except Exception as error:self.failed.emit(str(error))


class PasteCompositeWorker(QThread):
    done=pyqtSignal(object,object)
    failed=pyqtSignal(str)
    def __init__(self,layers,size):
        super().__init__();self.layers=layers;self.size=size
    def run(self):
        try:
            base=composite_layers(self.layers,self.size)
            scale=min(1.0,2048/max(base.size))
            preview=base.copy() if scale==1.0 else base.resize((max(1,round(base.width*scale)),max(1,round(base.height*scale))),R.LANCZOS,reducing_gap=2.0)
            self.done.emit(base,preview)
        except Exception as error:self.failed.emit(str(error))


class Canvas(QWidget):
    changed = pyqtSignal()
    zoom_requested = pyqtSignal(int, object)
    object_requested = pyqtSignal(object, int)
    point_requested = pyqtSignal(object, int)
    move_requested = pyqtSignal(int, int)
    layer_chosen = pyqtSignal(int)
    layer_resize_requested = pyqtSignal(int, int, int, int)
    transform_geometry_changed = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 350)
        self.setMouseTracking(True)
        self.image = self.original = self.marks = self.result = self.selection = None
        self.previous_selection = None
        self.revision = 0
        self.layers = []
        self.layer_index = -1
        self.canvas_size = None
        self.manual = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.tool = 'rectangle'
        self.selection_modifier = 0
        self.zoom = 100
        self.drag_start = self.drag_current = None
        self.drag_screen_start = None
        self.lasso_points = []
        self.history = []
        self.future = []
        self.history_budget = 512 * 1024 * 1024
        self.outline = None
        self.dash_offset = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_outline)
        self.timer.start(130)
        self.mode = 1
        self.brush = 28
        self.tolerance = 30
        self.last = None
        self.busy = False
        self.pix = None
        self.quick_lab = None
        self.transform = None
        self.move_preview = None
        self._move_pixmap_cache = None
        self._layer_preview_cache = {}
        self._layer_pixmap_cache = {}
        self._alpha_bbox_cache = {}
    def set_image(self, image):
        self.image = image
        self.original = image
        self.revision = 0
        self.layers = [{'name': '배경', 'image': image.copy(), 'x': 0, 'y': 0, 'visible': True}]
        self.layer_index = 0
        self.canvas_size = image.size
        self.quick_lab = None
        self.marks = Image.new('L', image.size, 0)
        self.result = self.selection = None
        self.history = []
        self.future = []
        self.refresh()
    def state(self):
        return (self.marks.copy(), self.selection.copy() if self.selection is not None else None,
                self.result.copy() if self.result is not None else None,
                self.image, self.original, self.revision,
                [layer.copy() for layer in self.layers], self.layer_index, self.canvas_size)
    def snapshot(self):
        if self.image is None: return
        self.history.append(self.state())
        self.history = self.history[-20:]
        self.future.clear()
        trim_history(self.history, self.future, self.history_budget)
    def undo(self):
        if not self.history or self.busy: return False
        self.future.append(self.state())
        self.marks, self.selection, self.result, self.image, self.original, self.revision, self.layers, self.layer_index, self.canvas_size = self.history.pop()
        trim_history(self.history, self.future, self.history_budget)
        self.quick_lab = None
        self.refresh()
        self.changed.emit()
        return True
    def redo(self):
        if not self.future or self.busy: return False
        self.history.append(self.state())
        self.marks, self.selection, self.result, self.image, self.original, self.revision, self.layers, self.layer_index, self.canvas_size = self.future.pop()
        trim_history(self.history, self.future, self.history_budget)
        self.quick_lab = None
        self.refresh()
        self.changed.emit()
        return True
    def animate_outline(self):
        if self.selection is not None or self.drag_start is not None:
            self.dash_offset = (self.dash_offset + 1) % 8
            self.update()
    def refresh(self):
        if self.image is not None:
            base = self.image.copy()
            if self.result is not None:
                # Preserve original alpha in the preview too.
                a = np.asarray(base.getchannel('A'), dtype=np.uint16)
                b = np.asarray(self.result.resize(base.size, R.LANCZOS), dtype=np.uint16)
                base.putalpha(Image.fromarray((a*b//255).astype('uint8')))
            self.pix = QPixmap.fromImage(qimage(base))
            marks = np.asarray(self.marks)
            overlay = np.zeros((*marks.shape, 4), dtype=np.uint8)
            self.outline = None
            if self.selection is not None:
                mask = (np.asarray(self.selection.resize(self.image.size, R.NEAREST)) >= 128).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                path = QPainterPath()
                for contour in contours:
                    pts = contour.reshape(-1, 2)
                    if len(pts) < 2: continue
                    path.moveTo(float(pts[0,0]), float(pts[0,1]))
                    for x, y in pts[1:]: path.lineTo(float(x), float(y))
                    path.closeSubpath()
                self.outline = path
            else:
                overlay[marks == 1] = (31, 222, 127, 51)
                overlay[marks == 2] = (255, 73, 86, 51)
            self.overlay = QPixmap.fromImage(qimage(Image.fromarray(overlay)))
        self.update()
    def image_rect(self):
        if self.image is None: return QRectF()
        w, h = self.image.size
        scale = min((self.width()-48)/w, (self.height()-48)/h)
        dw, dh = w*scale, h*scale
        return QRectF((self.width()-dw)/2, (self.height()-dh)/2, dw, dh)
    def transform_view(self):
        state = self.transform
        sx, sy, sw, sh = state['scene']
        scale = min(max(1,self.width()-72)/sw, max(1,self.height()-72)/sh)
        ox = (self.width()-sw*scale)/2 - sx*scale
        oy = (self.height()-sh*scale)/2 - sy*scale
        return ox, oy, scale
    def transform_screen_rect(self):
        ox,oy,scale = self.transform_view()
        x,y,w,h = self.transform['rect']
        return QRectF(ox+x*scale,oy+y*scale,w*scale,h*scale)
    def transform_handles(self):
        r = self.transform_screen_rect()
        return {'nw':r.topLeft(), 'n':QPointF(r.center().x(),r.top()), 'ne':r.topRight(),
                'e':QPointF(r.right(),r.center().y()), 'se':r.bottomRight(),
                's':QPointF(r.center().x(),r.bottom()), 'sw':r.bottomLeft(), 'w':QPointF(r.left(),r.center().y())}
    def transform_hit(self, point):
        for name, pos in self.transform_handles().items():
            if abs(point.x()-pos.x()) <= 9 and abs(point.y()-pos.y()) <= 9: return name
        if self.transform_screen_rect().contains(point): return 'move'
        return None
    def transform_cursor(self, hit):
        cursors={'nw':Qt.SizeFDiagCursor,'se':Qt.SizeFDiagCursor,'ne':Qt.SizeBDiagCursor,'sw':Qt.SizeBDiagCursor,
                 'n':Qt.SizeVerCursor,'s':Qt.SizeVerCursor,'e':Qt.SizeHorCursor,'w':Qt.SizeHorCursor,'move':Qt.SizeAllCursor}
        self.setCursor(cursors.get(hit,Qt.ArrowCursor))
    def paint_transform(self, p):
        state = self.transform
        ox,oy,scale = self.transform_view()
        cw,ch = self.canvas_size
        canvas_rect = QRectF(ox,oy,cw*scale,ch*scale)
        p.save(); p.setClipRect(canvas_rect)
        tile=14; left,top=int(canvas_rect.left()),int(canvas_rect.top())
        for yy in range(top,int(canvas_rect.bottom())+1,tile):
            for xx in range(left,int(canvas_rect.right())+1,tile):
                p.fillRect(xx,yy,tile,tile,QColor('#e2e5eb' if ((xx-left)//tile+(yy-top)//tile)%2==0 else '#bec4cf'))
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(canvas_rect,state['underlay'],QRectF(state['underlay'].rect()))
        p.restore()
        box=self.transform_screen_rect()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(box,state['pixmap'],QRectF(state['pixmap'].rect()))
        p.save();p.setClipRect(canvas_rect)
        p.drawPixmap(canvas_rect,state['above'],QRectF(state['above'].rect()))
        p.restore()
        p.setPen(QPen(QColor('#87a9ff'),1.5));p.setBrush(Qt.NoBrush);p.drawRect(box)
        p.setBrush(QColor('#f6f8ff'));p.setPen(QPen(QColor('#526ff0'),1))
        for pos in self.transform_handles().values():p.drawRect(QRectF(pos.x()-4,pos.y()-4,8,8))
    def transform_press(self,event):
        hit=self.transform_hit(event.localPos());self.transform_cursor(hit)
        if hit is not None:
            ox,oy,scale=self.transform_view()
            self.transform['move_axis']=None
            self.transform['drag']=(hit, event.localPos(), tuple(self.transform['rect']), scale)
        event.accept()
    def transform_move(self,event):
        drag=self.transform.get('drag')
        if drag is None:
            self.transform_cursor(self.transform_hit(event.localPos()));return
        hit,start,original,scale=drag
        dx=(event.localPos().x()-start.x())/scale;dy=(event.localPos().y()-start.y())/scale
        x,y,w,h=original
        if hit=='move':
            if event.modifiers() & Qt.ShiftModifier:
                if self.transform.get('move_axis') is None and max(abs(dx),abs(dy)) >= 1:
                    self.transform['move_axis']='x' if abs(dx) >= abs(dy) else 'y'
                if self.transform.get('move_axis') == 'x': dy=0
                elif self.transform.get('move_axis') == 'y': dx=0
            else: self.transform['move_axis']=None
            new=[x+dx,y+dy,w,h]
        else:
            centered=bool(event.modifiers() & Qt.AltModifier)
            left,top,right,bottom=x,y,x+w,y+h
            if 'w' in hit:
                left=min(x+w/2-0.5 if centered else right-1,left+dx)
                if centered:right=2*(x+w/2)-left
            if 'e' in hit:
                right=max(x+w/2+0.5 if centered else left+1,right+dx)
                if centered:left=2*(x+w/2)-right
            if 'n' in hit:
                top=min(y+h/2-0.5 if centered else bottom-1,top+dy)
                if centered:bottom=2*(y+h/2)-top
            if 's' in hit:
                bottom=max(y+h/2+0.5 if centered else top+1,bottom+dy)
                if centered:top=2*(y+h/2)-bottom
            nw,nh=right-left,bottom-top
            locked=bool(self.transform.get('keep_ratio',True)) != bool(event.modifiers() & Qt.ShiftModifier)
            if locked:
                sx,sy=nw/w,nh/h
                if hit in ('e','w'):factor=sx
                elif hit in ('n','s'):factor=sy
                else:factor=sx if abs(sx-1)>=abs(sy-1) else sy
                factor=max(max(1/w,1/h),min(factor,12000/max(w,h)))
                nw,nh=w*factor,h*factor
                if centered:left=x+(w-nw)/2;top=y+(h-nh)/2
                else:
                    left=x+w-nw if 'w' in hit else x if 'e' in hit else x+(w-nw)/2
                    top=y+h-nh if 'n' in hit else y if 's' in hit else y+(h-nh)/2
            else:
                nw=min(12000,nw);nh=min(12000,nh)
                if centered:left=x+(w-nw)/2;top=y+(h-nh)/2
                else:
                    if 'w' in hit:left=x+w-nw
                    if 'n' in hit:top=y+h-nh
            new=[left,top,nw,nh]
        self.transform['rect']=new;self.update();self.transform_geometry_changed.emit()
    def layer_alpha_bounds(self, image):
        key=id(image); cached=self._alpha_bbox_cache.get(key)
        if cached is not None and cached[0] is image: return cached[1]
        if len(self._alpha_bbox_cache) >= 8: self._alpha_bbox_cache.clear()
        bounds=image.getchannel('A').getbbox()
        self._alpha_bbox_cache[key]=(image,bounds)
        return bounds
    def move_layer_screen_rect(self):
        if self.canvas_size is None or self.image is None or self.layer_index < 0 or self.layer_index >= len(self.layers): return None
        layer=self.layers[self.layer_index]
        if not layer['visible']: return None
        bounds=self.layer_alpha_bounds(layer['image'])
        if bounds is None:return None
        if self.move_preview is not None:
            state=self.move_preview;x,y=state['newx'],state['newy']
            sx,sy=state['neww']/state['w'],state['newh']/state['h']
        else:x,y=layer['x'],layer['y'];sx=sy=1.0
        r=self.image_rect();cw,ch=self.canvas_size
        return QRectF(r.left()+(x+bounds[0]*sx)*r.width()/cw,
                      r.top()+(y+bounds[1]*sy)*r.height()/ch,
                      (bounds[2]-bounds[0])*sx*r.width()/cw,
                      (bounds[3]-bounds[1])*sy*r.height()/ch)
    def move_handles(self):
        r=self.move_layer_screen_rect()
        if r is None:return {}
        return {'nw':r.topLeft(),'n':QPointF(r.center().x(),r.top()),'ne':r.topRight(),
                'e':QPointF(r.right(),r.center().y()),'se':r.bottomRight(),
                's':QPointF(r.center().x(),r.bottom()),'sw':r.bottomLeft(),'w':QPointF(r.left(),r.center().y())}
    def move_handle_at(self, point):
        for name,pos in self.move_handles().items():
            if abs(point.x()-pos.x())<=8 and abs(point.y()-pos.y())<=8:return name
        return None
    def layer_at(self, point):
        p=self.point(point)
        if p is None or self.canvas_size is None:return -1
        cx=int(p[0]*self.canvas_size[0]/self.image.width)
        cy=int(p[1]*self.canvas_size[1]/self.image.height)
        for index in range(len(self.layers)-1,-1,-1):
            layer=self.layers[index]
            if not layer['visible']:continue
            image=layer['image'];x,y=cx-layer['x'],cy-layer['y']
            if 0<=x<image.width and 0<=y<image.height and image.getpixel((x,y))[3]>0:return index
        return -1
    def update_move_preview(self,event):
        state=self.move_preview;r=self.image_rect();start=state['start'];cw,ch=self.canvas_size
        dx=(event.localPos().x()-start.x())*cw/r.width();dy=(event.localPos().y()-start.y())*ch/r.height()
        if state['kind']=='move':
            if event.modifiers() & Qt.ShiftModifier:
                if state.get('axis') is None and max(abs(dx),abs(dy)) >= 1:
                    state['axis']='x' if abs(dx) >= abs(dy) else 'y'
                if state.get('axis') == 'x': dy=0
                elif state.get('axis') == 'y': dx=0
            else: state['axis']=None
            state['newx']=round(state['x']+dx);state['newy']=round(state['y']+dy)
        else:
            hit=state['kind'];bx0,by0,bx1,by1=state['bounds'];x,y,w,h=state['x'],state['y'],state['w'],state['h']
            left,top,right,bottom=x+bx0,y+by0,x+bx1,y+by1
            if 'w' in hit:left=min(right-1,left+dx)
            if 'e' in hit:right=max(left+1,right+dx)
            if 'n' in hit:top=min(bottom-1,top+dy)
            if 's' in hit:bottom=max(top+1,bottom+dy)
            sx=(right-left)/(bx1-bx0);sy=(bottom-top)/(by1-by0)
            if len(hit)==2 and not (event.modifiers() & Qt.ShiftModifier):
                factor=sx if abs(sx-1)>=abs(sy-1) else sy
                factor=max(1/max(w,h),min(factor,12000/max(w,h)))
                sx=sy=factor
                if 'w' in hit:left=x+bx1-(bx1-bx0)*sx
                if 'n' in hit:top=y+by1-(by1-by0)*sy
            state['neww']=max(1,min(12000,round(w*sx)));state['newh']=max(1,min(12000,round(h*sy)))
            state['newx']=round(left-bx0*sx);state['newy']=round(top-by0*sy)
        self.update()
    def paint_move_bounds(self,p):
        r=self.move_layer_screen_rect()
        if r is None:return
        p.save();p.setPen(QPen(QColor('#87a9ff'),1.5));p.setBrush(Qt.NoBrush);p.drawRect(r)
        p.setBrush(QColor('#f6f8ff'));p.setPen(QPen(QColor('#526ff0'),1))
        for pos in self.move_handles().values():p.drawRect(QRectF(pos.x()-4,pos.y()-4,8,8))
        p.restore()
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#dce3ed' if getattr(self, 'light_theme', False) else '#20242e'))
        if self.transform is not None:
            self.paint_transform(p); return
        if self.image is None:
            p.setPen(QColor('#53667d' if getattr(self, 'light_theme', False) else '#c8cfdd'))
            p.drawText(self.rect(), Qt.AlignCenter, '사진을 열어 누끼 작업을 시작하세요')
            return
        rect = self.image_rect()
        p.save()
        p.setClipRect(rect)
        size = 16
        left, top = int(rect.left()), int(rect.top())
        visible=rect.intersected(QRectF(event.rect()))
        start_y=top+max(0,int((visible.top()-top)//size))*size
        start_x=left+max(0,int((visible.left()-left)//size))*size
        for y in range(start_y, int(visible.bottom())+1, size):
            for x in range(start_x, int(visible.right())+1, size):
                p.fillRect(x, y, size, size, QColor('#e2e5eb' if ((x-left)//size+(y-top)//size)%2 == 0 else '#bec4cf'))
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        if self.move_preview is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            state=self.move_preview; p.drawPixmap(rect,state['underlay'],QRectF(state['underlay'].rect()))
            x,y=state['newx'],state['newy'];cw,ch=self.canvas_size
            layer_rect=QRectF(rect.left()+x*rect.width()/cw,rect.top()+y*rect.height()/ch,
                              state['neww']*rect.width()/cw,state['newh']*rect.height()/ch)
            p.drawPixmap(layer_rect,state['pixmap'],QRectF(state['pixmap'].rect()))
            p.drawPixmap(rect,state['above'],QRectF(state['above'].rect()))
        else:
            p.drawPixmap(rect, self.pix, QRectF(self.pix.rect()))
            p.drawPixmap(rect, self.overlay, QRectF(self.overlay.rect()))
        if self.tool == 'move': self.paint_move_bounds(p)
        if self.drag_start is not None and self.drag_current is not None:
            if self.tool == 'lasso' and len(self.lasso_points) > 1:
                path = QPainterPath()
                x,y = self.lasso_points[0]; path.moveTo(rect.left()+x*rect.width()/self.image.width, rect.top()+y*rect.height()/self.image.height)
                for x,y in self.lasso_points[1:]: path.lineTo(rect.left()+x*rect.width()/self.image.width, rect.top()+y*rect.height()/self.image.height)
                pen = QPen(QColor('black')); pen.setCosmetic(True); p.setPen(pen); p.drawPath(path)
                pen = QPen(QColor('white')); pen.setCosmetic(True); pen.setDashPattern([4,4]); pen.setDashOffset(self.dash_offset); p.setPen(pen); p.drawPath(path)
            x1, y1 = self.drag_start; x2, y2 = self.drag_current
            box = QRectF(rect.left() + x1*rect.width()/self.image.width, rect.top() + y1*rect.height()/self.image.height,
                         (x2-x1)*rect.width()/self.image.width, (y2-y1)*rect.height()/self.image.height).normalized()
            if self.tool != 'lasso':
                pen = QPen(QColor('black')); pen.setCosmetic(True); p.setPen(pen); p.drawRect(box)
                pen = QPen(QColor('white')); pen.setCosmetic(True); pen.setDashPattern([4,4]); pen.setDashOffset(self.dash_offset); p.setPen(pen); p.drawRect(box)
        if self.outline is not None:
            p.save()
            p.translate(rect.left(), rect.top())
            p.scale(rect.width() / self.image.width, rect.height() / self.image.height)
            pen = QPen(QColor('black')); pen.setWidthF(1); pen.setCosmetic(True)
            p.setPen(pen); p.drawPath(self.outline)
            pen = QPen(QColor('white')); pen.setWidthF(1); pen.setCosmetic(True)
            pen.setDashPattern([4, 4]); pen.setDashOffset(self.dash_offset)
            p.setPen(pen); p.drawPath(self.outline)
            p.restore()
        p.restore()
    def point(self, pos):
        r = self.image_rect()
        if not r.contains(pos): return None
        return ((pos.x()-r.left())*self.image.width/r.width(), (pos.y()-r.top())*self.image.height/r.height())
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            self.zoom_requested.emit(1 if event.angleDelta().y() > 0 else -1, event.pos())
            event.accept()
        else:
            event.ignore()
    def update_cursor(self, modifiers):
        if self.tool == 'move':
            self.setCursor(Qt.SizeAllCursor)
        elif self.tool in ('rectangle', 'lasso', 'object', 'wand'):
            pix = QPixmap(32, 32); pix.fill(Qt.transparent)
            painter = QPainter(pix); painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor('black'), 3)); painter.drawLine(2, 12, 22, 12); painter.drawLine(12, 2, 12, 22)
            painter.setPen(QPen(QColor('white'), 1)); painter.drawLine(2, 12, 22, 12); painter.drawLine(12, 2, 12, 22)
            symbol = '+' if (modifiers | self.selection_modifier) & Qt.ShiftModifier else '−' if (modifiers | self.selection_modifier) & Qt.AltModifier else ''
            if symbol:
                painter.setPen(QColor('white')); painter.drawText(20, 27, symbol)
            painter.end(); self.setCursor(QCursor(pix, 12, 12))
        elif self.manual and self.tool == 'brush':
            diameter = max(8, min(120, int(self.brush)))
            size = diameter + 6; center = size // 2
            pix = QPixmap(size, size); pix.fill(Qt.transparent)
            p = QPainter(pix); p.setRenderHint(QPainter.Antialiasing)
            outline = QRectF(3, 3, diameter-1, diameter-1)
            p.setPen(QPen(QColor('black'), 3)); p.drawEllipse(outline)
            p.setPen(QPen(QColor('white'), 1)); p.drawEllipse(outline)
            p.setPen(QPen(QColor('black'), 3)); p.drawLine(center-3, center, center+3, center)
            if not modifiers & Qt.AltModifier: p.drawLine(center, center-3, center, center+3)
            p.setPen(QPen(QColor('white'), 1)); p.drawLine(center-3, center, center+3, center)
            if not modifiers & Qt.AltModifier: p.drawLine(center, center-3, center, center+3)
            p.end(); self.setCursor(QCursor(pix, center, center))
        elif self.manual:
            self.setCursor(QCursor(Qt.CrossCursor))
        else: self.setCursor(QCursor(Qt.ArrowCursor))
    def mousePressEvent(self, event):
        if self.transform is not None:
            if event.button() == Qt.LeftButton: self.transform_press(event)
            return
        if self.image is None or self.busy or event.button() != Qt.LeftButton: return
        self.setFocus(Qt.MouseFocusReason)
        if self.tool == 'move':
            if self.point(event.localPos()) is None: return
            handle=self.move_handle_at(event.localPos())
            if handle is None:
                index=self.layer_at(event.localPos())
                if index < 0: return
                if index != self.layer_index:
                    self.layer_index=index;self.layer_chosen.emit(index)
            if self.layer_index < 0:return
            layer=self.layers[self.layer_index]
            if not layer['visible']:return
            bounds=self.layer_alpha_bounds(layer['image'])
            if bounds is None:return
            cache_key=(self.revision,self.layer_index,self.canvas_size,tuple((id(item['image']),item['x'],item['y'],item['visible']) for item in self.layers))
            if self._move_pixmap_cache is None or self._move_pixmap_cache[0] != cache_key:
                image=layer['image'];pixkey=id(image);pixcache=self._layer_pixmap_cache.get(pixkey)
                if pixcache is None or pixcache[0] is not image:
                    if len(self._layer_pixmap_cache)>=8:self._layer_pixmap_cache.pop(next(iter(self._layer_pixmap_cache)))
                    pixcache=(image,preview_pixmap(image));self._layer_pixmap_cache[pixkey]=pixcache
                self._move_pixmap_cache=(cache_key,
                    composite_preview(self.layers[:self.layer_index],self.canvas_size,cache=self._layer_preview_cache),
                    composite_preview(self.layers[self.layer_index+1:],self.canvas_size,cache=self._layer_preview_cache),
                    pixcache[1])
            _,underlay,above,layer_pixmap=self._move_pixmap_cache
            self.move_preview={'start':event.localPos(), 'kind':handle or 'move', 'bounds':bounds,
                'x':layer['x'], 'y':layer['y'], 'w':layer['image'].width, 'h':layer['image'].height,
                'newx':layer['x'], 'newy':layer['y'], 'neww':layer['image'].width, 'newh':layer['image'].height,
                'underlay':underlay, 'above':above, 'pixmap':layer_pixmap}
            self.update();event.accept();return
        point = self.point(event.localPos())
        if point is None:
            if self.tool in ('rectangle', 'object', 'lasso') and self.selection is not None and not (event.modifiers() | self.selection_modifier) & (Qt.ShiftModifier | Qt.AltModifier):
                self.snapshot(); self.selection = None
                self.refresh(); self.changed.emit()
            return
        if self.tool == 'wand':
            self.point_requested.emit(point, int(event.modifiers())); return
        if self.tool in ('rectangle', 'object', 'lasso'):
            self.drag_start = self.drag_current = point
            self.drag_screen_start = event.pos()
            self.lasso_points = [point]
            self.drag_modifiers = event.modifiers() | self.selection_modifier
            self.update()
        elif self.manual and self.tool == 'brush':
            self.snapshot()
            self.quick_modifiers = event.modifiers()
            # A plain stroke starts a new selection; only Shift adds to the
            # prior selection and Alt subtracts from it. Keep accumulating
            # within the same stroke so dragging selects adjacent regions.
            if not self.quick_modifiers & (Qt.ShiftModifier | Qt.AltModifier):
                self.selection = None
            self.last = point
            self.quick_select(point)
    def mouseMoveEvent(self, event):
        if self.transform is not None:
            self.transform_move(event); return
        self.update_cursor(event.modifiers())
        if self.move_preview is not None and event.buttons() & Qt.LeftButton:
            self.update_move_preview(event);return
        if self.tool == 'move':
            self.transform_cursor(self.move_handle_at(event.localPos()) or 'move')
        if self.image is None or self.busy: return
        if self.drag_start is not None and event.buttons() & Qt.LeftButton:
            rect = self.image_rect(); pos = event.localPos()
            self.drag_current = (max(0, min(self.image.width-1, (pos.x()-rect.left())*self.image.width/rect.width())),
                                 max(0, min(self.image.height-1, (pos.y()-rect.top())*self.image.height/rect.height())))
            if self.tool == 'lasso': self.lasso_points.append(self.drag_current)
            self.update()
        elif event.buttons() & Qt.LeftButton and self.manual and self.tool == 'brush':
            point = self.point(event.localPos())
            if point is not None: self.quick_select(point)
    def mouseReleaseEvent(self, event):
        if self.transform is not None:
            if self.transform.get('drag') is not None:
                self.transform_move(event)
                original=self.transform['drag'][2]
                if any(abs(a-b)>0.001 for a,b in zip(original,self.transform['rect'])):
                    self.transform['undo'].append(list(original));self.transform['redo'].clear()
                self.transform['drag']=None
                self.transform_geometry_changed.emit()
            return
        if self.move_preview is not None:
            self.update_move_preview(event);state=self.move_preview
            self.move_preview=None;self.update()
            if state['kind']=='move':
                dx,dy=state['newx']-state['x'],state['newy']-state['y']
                if dx or dy:self.move_requested.emit(dx,dy)
            elif (state['newx'],state['newy'],state['neww'],state['newh']) != (state['x'],state['y'],state['w'],state['h']):
                self.layer_resize_requested.emit(state['newx'],state['newy'],state['neww'],state['newh'])
            return
        if self.drag_start is not None:
            from PIL import ImageDraw
            endpoint = self.point(event.localPos())
            if endpoint is not None: self.drag_current = endpoint
            x1, y1 = self.drag_start; x2, y2 = self.drag_current
            self.drag_start = self.drag_current = None
            is_click = self.drag_screen_start is not None and (event.pos() - self.drag_screen_start).manhattanLength() < QApplication.startDragDistance()
            self.drag_screen_start = None
            if is_click:
                if self.selection is not None and not self.drag_modifiers & (Qt.ShiftModifier | Qt.AltModifier) and self.selection.getpixel((int(x1), int(y1))) < 128:
                    self.snapshot(); self.selection = None
                    self.refresh(); self.changed.emit()
                else: self.update()
            elif self.tool == 'object' and abs(x2-x1) >= 1 and abs(y2-y1) >= 1:
                self.object_requested.emit((min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)), int(self.drag_modifiers))
            elif self.tool == 'lasso' and len(self.lasso_points) >= 3:
                mask = Image.new('L', self.image.size, 0)
                ImageDraw.Draw(mask).polygon(self.lasso_points, fill=255)
                self.apply_selection(mask, self.drag_modifiers)
            elif self.tool == 'rectangle' and abs(x2-x1) >= 1 and abs(y2-y1) >= 1:
                self.snapshot()
                mods = self.drag_modifiers
                if mods & (Qt.ShiftModifier | Qt.AltModifier) and self.selection is not None:
                    selection = self.selection.copy()
                else:
                    selection = Image.new('L', self.image.size, 0)
                ImageDraw.Draw(selection).rectangle((min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)),
                                                     fill=0 if mods & Qt.AltModifier else 255)
                self.selection = selection
                self.refresh(); self.changed.emit()
            else: self.update()
        if self.tool == 'brush' and self.manual and self.last is not None:
            endpoint = self.point(event.localPos())
            if endpoint is not None: self.quick_select(endpoint)
        self.last = None
        self.lasso_points = []
    def apply_selection(self, mask, mods):
        # Mouse signals carry an int; direct calls may pass Qt.KeyboardModifiers.
        # Normalize both before bitwise tests to avoid a PyQt type mismatch.
        flags = int(mods)
        shift = bool(flags & int(Qt.ShiftModifier))
        alt = bool(flags & int(Qt.AltModifier))
        self.snapshot()
        incoming = np.asarray(mask.convert('L'), dtype=np.uint8)
        if self.selection is not None and (shift or alt):
            previous = np.asarray(self.selection, dtype=np.uint8)
            incoming = np.minimum(previous, 255-incoming) if alt else np.maximum(previous, incoming)
        elif alt:
            incoming = np.zeros_like(incoming)
        self.selection = Image.fromarray(incoming)
        self.refresh(); self.changed.emit()
    def quick_select(self, point):
        # Direct selection: find connected, similarly colored areas at the
        # cursor and between pointer events, then update the selection outline.
        if self.quick_lab is None:
            rgb = np.asarray(self.image.convert('RGB'))
            self.quick_lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (3,3), 0), cv2.COLOR_RGB2LAB)
        h, w = self.quick_lab.shape[:2]
        previous_point = self.last if self.last is not None else point
        distance = ((point[0]-previous_point[0])**2 + (point[1]-previous_point[1])**2)**0.5
        display_scale = self.image_rect().width() / w
        radius = max(1.0, self.brush / (2.0 * max(display_scale, 0.001)))
        steps = min(80, max(1, int(distance / max(2.0, radius * 0.7)) + 1))
        mask = np.zeros((h+2, w+2), np.uint8)
        flags = 8 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
        # Sample throughout the circle, so increasing its diameter changes
        # which nearby regions are picked up instead of merely changing the cursor.
        offsets = [(0.0, 0.0)]
        for ring in (0.45, 0.82):
            for k in range(8):
                angle = 2 * np.pi * k / 8
                offsets.append((radius * ring * np.cos(angle), radius * ring * np.sin(angle)))
        # Larger tolerance accepts a wider local color difference in Lab space.
        value = max(0, min(100, int(self.tolerance)))
        delta_l = max(2, round(2 + value * 0.6))
        delta_ab = max(2, round(2 + value * 0.35))
        tolerance = (delta_l, delta_ab, delta_ab)
        for i in range(steps + 1):
            t = i / steps
            cx = previous_point[0] + (point[0]-previous_point[0])*t
            cy = previous_point[1] + (point[1]-previous_point[1])*t
            for dx, dy in offsets:
                x = max(0, min(w-1, round(cx + dx)))
                y = max(0, min(h-1, round(cy + dy)))
                if mask[y+1, x+1] == 0:
                    cv2.floodFill(self.quick_lab, mask, (x,y), (0,0,0), tolerance, tolerance, flags)
        added = mask[1:-1,1:-1]
        previous = np.asarray(self.selection, dtype=np.uint8) if self.selection is not None else np.zeros((h,w),dtype=np.uint8)
        result = np.minimum(previous, 255-added) if self.quick_modifiers & Qt.AltModifier else np.maximum(previous, added)
        self.selection = Image.fromarray(result)
        self.last = point
        self.refresh(); self.changed.emit()


class Navigator(QLabel):
    pan_requested = pyqtSignal(float, float)
    def __init__(self):
        super().__init__('사진 없음')
        self.view_rect = QRectF()
        self.drag_offset = None
    def preview_origin(self):
        pix = self.pixmap()
        return QPointF((self.width()-pix.width())/2, (self.height()-pix.height())/2) if pix else QPointF()
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.pixmap() is None: return
        self.setCursor(Qt.ClosedHandCursor)
        point = event.localPos() - self.preview_origin()
        if self.view_rect.contains(point):
            self.drag_offset = point - self.view_rect.topLeft()
        else:
            self.drag_offset = QPointF(self.view_rect.width()/2, self.view_rect.height()/2)
        self.pan_requested.emit(point.x()-self.drag_offset.x(), point.y()-self.drag_offset.y())
    def mouseMoveEvent(self, event):
        if self.drag_offset is not None and event.buttons() & Qt.LeftButton:
            point = event.localPos() - self.preview_origin() - self.drag_offset
            self.pan_requested.emit(point.x(), point.y())
    def mouseReleaseEvent(self, event):
        self.drag_offset = None
        self.setCursor(Qt.OpenHandCursor if self.pixmap() is not None else Qt.ArrowCursor)


def tool_icon(kind, color='#e6edf8'):
    # Lucide icons are redistributed under the ISC license in assets/lucide.
    names = {'save':'save', 'undo':'undo-2', 'redo':'redo-2',
             'rectangle':'square-dashed', 'lasso':'lasso', 'plus':'lasso-select',
             'minus':'lasso-select', 'object':'scan-search', 'brush':'brush',
             'wand':'wand-sparkles', 'subject':'scan-face', 'close':'x', 'move':'move'}
    asset = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'lucide', names[kind] + '.svg')
    with open(asset, 'rb') as f: svg = f.read().replace(b'currentColor', color.encode('ascii'))
    renderer = QSvgRenderer(QByteArray(svg))
    if not renderer.isValid(): raise ValueError('아이콘 파일을 읽을 수 없습니다: ' + asset)
    pix = QPixmap(48,48); pix.fill(Qt.transparent)
    p = QPainter(pix); p.setRenderHint(QPainter.Antialiasing)
    if kind in ('plus', 'minus'):
        renderer.render(p, QRectF(0,0,38,38))
        badge_name = 'plus' if kind == 'plus' else 'minus'
        badge_path = os.path.join(os.path.dirname(asset), badge_name + '.svg')
        with open(badge_path, 'rb') as f: badge_svg = f.read().replace(b'currentColor', b'#a9c0ff')
        QSvgRenderer(QByteArray(badge_svg)).render(p, QRectF(28,28,20,20))
    else:
        renderer.render(p, QRectF(2,2,44,44))
    p.end(); return QIcon(pix)


class AnimatedButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.highlight = 0.0
        self.transition = QVariantAnimation(self)
        self.transition.setDuration(140)
        self.transition.setEasingCurve(QEasingCurve.OutCubic)
        self.transition.valueChanged.connect(self.set_highlight)
    def set_highlight(self, value):
        self.highlight = float(value)
        self.update()
    def animate_to(self, value):
        self.transition.stop()
        self.transition.setStartValue(self.highlight)
        self.transition.setEndValue(value)
        self.transition.start()
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.highlight > 0.001:
            painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, round(255 * self.highlight)))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 5, 5)
            painter.end()
    def enterEvent(self, event):
        if self.isEnabled(): self.animate_to(0.10)
        super().enterEvent(event)
    def leaveEvent(self, event):
        self.animate_to(0.0)
        super().leaveEvent(event)
    def mousePressEvent(self, event):
        if self.isEnabled(): self.animate_to(0.18)
        super().mousePressEvent(event)
    def mouseReleaseEvent(self, event):
        self.animate_to(0.10 if self.rect().contains(event.pos()) and self.isEnabled() else 0.0)
        super().mouseReleaseEvent(event)
    def changeEvent(self, event):
        if not self.isEnabled() and hasattr(self, 'transition'): self.animate_to(0.0)
        super().changeEvent(event)


class ModeSwitch(QAbstractButton):
    """One switch: click to toggle or drag the thumb to either side."""
    def __init__(self):
        super().__init__()
        self.setCheckable(True)
        self.setFixedSize(190, 34)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip('클릭하거나 좌우로 드래그하여 자동/수동 모드를 전환합니다')
        self.dragging = False
        self._position = 0.0
        self.animation = QPropertyAnimation(self, b'position', self)
        self.animation.setDuration(220)
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.toggled.connect(self.animate_switch)
    @pyqtProperty(float)
    def position(self): return self._position
    @position.setter
    def position(self, value):
        self._position = value
        self.update()
    def animate_switch(self, checked):
        self.animation.stop()
        self.animation.setStartValue(self.position)
        self.animation.setEndValue(1.0 if checked else 0.0)
        self.animation.start()
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        light=getattr(self,'light_theme',False)
        p.setPen(QPen(QColor('#b2c0d0' if light else '#566274'), 1)); p.setBrush(QColor('#e4ebf4' if light else '#252b35'))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)
        x = 2 + self._position * (self.width() // 2 - 2)
        p.setPen(Qt.NoPen); p.setBrush(QColor('#526ff0'))
        p.drawRoundedRect(QRectF(x, 2, self.width() // 2 - 2, self.height() - 4), 6, 6)
        p.setPen(QColor('#ffffff') if not self.isChecked() else QColor('#53667d' if light else '#cbd3df'))
        p.drawText(QRectF(0, 0, self.width()/2, self.height()), Qt.AlignCenter, 'Auto' if getattr(self,'english',False) else '자동 모드')
        p.setPen(QColor('#ffffff') if self.isChecked() else QColor('#53667d' if light else '#cbd3df'))
        p.drawText(QRectF(self.width()/2, 0, self.width()/2, self.height()), Qt.AlignCenter, 'Manual' if getattr(self,'english',False) else '수동 모드')
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True; self.start_x = event.pos().x(); event.accept()
        else: super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self.dragging:
            self.animation.stop()
            self.position = max(0.0, min(1.0, (event.pos().x() - self.width()/4) / (self.width()/2)))
            event.accept()
        else: super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event):
        if self.dragging and event.button() == Qt.LeftButton:
            self.dragging = False
            if abs(event.pos().x() - self.start_x) > 5:
                target = event.pos().x() >= self.width()/2
            else:
                target = not self.isChecked()
            if target == self.isChecked(): self.animate_switch(target)
            else: self.setChecked(target)
            event.accept()
        else: super().mouseReleaseEvent(event)


class PptExportDialog(QDialog):
    def __init__(self, documents, parent=None):
        super().__init__(parent); self.setWindowFlag(Qt.WindowContextHelpButtonHint, False);self.setWindowTitle('PPT로 내보내기');self.setMinimumWidth(420)
        self.setStyleSheet('''
            QDialog { background:#1d2330; color:#eef2fa; }
            QLabel { background:transparent; color:#e7edf7; }
            QListWidget, QComboBox { background:#2b3545; color:#f5f7fb; border:1px solid #536075; border-radius:5px; padding:5px; }
            QListWidget::item { padding:7px; } QListWidget::item:selected { background:#42536e; }
            QComboBox QAbstractItemView { background:#2b3545; color:#f5f7fb; selection-background-color:#526ff0; }
            QPushButton { background:#354155; color:#f5f7fb; border:1px solid #536075; border-radius:6px; padding:8px 15px; }
            QPushButton:hover { background:#44536b; } QPushButton#exportPpt { background:#526ff0; border:none; font-weight:bold; }
        ''')
        layout=QVBoxLayout(self);layout.setContentsMargins(24,22,24,22);layout.setSpacing(10)
        title=QLabel('PPT로 내보내기');title.setStyleSheet('font-size:20px; font-weight:bold;');layout.addWidget(title)
        layout.addWidget(QLabel('슬라이드로 만들 탭 선택'))
        self.tab_list=QListWidget();self.tab_list.setMinimumHeight(150)
        for index,name in documents:
            item=QListWidgetItem(name);item.setData(Qt.UserRole,index)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable);item.setCheckState(Qt.Checked);self.tab_list.addItem(item)
        layout.addWidget(self.tab_list)
        choices=QHBoxLayout()
        all_btn=QPushButton('전체 선택');all_btn.clicked.connect(lambda:self.set_checked(True));choices.addWidget(all_btn)
        none_btn=QPushButton('선택 해제');none_btn.clicked.connect(lambda:self.set_checked(False));choices.addWidget(none_btn)
        choices.addStretch();layout.addLayout(choices)
        layout.addWidget(QLabel('슬라이드 크기'))
        self.slide_size=QComboBox();self.slide_size.addItem('와이드 16:9 (13.33 × 7.5인치)','wide');self.slide_size.addItem('표준 4:3 (10 × 7.5인치)','standard');layout.addWidget(self.slide_size)
        info=QLabel('탭 순서대로 한 장씩 만듭니다. 보이는 레이어는 PPT의 개별 이미지로 들어갑니다.');info.setWordWrap(True);info.setStyleSheet('color:#b5c2d4; font-size:12px;');layout.addWidget(info)
        row=QHBoxLayout();row.addStretch();cancel=QPushButton('취소');cancel.clicked.connect(self.reject);row.addWidget(cancel)
        export=QPushButton('내보내기');export.setObjectName('exportPpt');export.clicked.connect(self.validate_selection);row.addWidget(export);layout.addLayout(row)
    def set_checked(self, checked):
        for i in range(self.tab_list.count()):self.tab_list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)
    def selected_indices(self):
        return [int(self.tab_list.item(i).data(Qt.UserRole)) for i in range(self.tab_list.count()) if self.tab_list.item(i).checkState()==Qt.Checked]
    def validate_selection(self):
        if not self.selected_indices():
            QMessageBox.information(self,'탭 선택','PPT에 넣을 탭을 하나 이상 선택해 주세요.');return
        self.accept()


def suggested_pptx_filename(tab_name):
    name=str(tab_name or '').strip()
    stem,ext=os.path.splitext(name)
    if ext.lower() in ('.png','.jpg','.jpeg','.webp','.bmp','.heic','.heif','.gif','.pptx'):
        name=stem
    name=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',name).strip(' .') or '제목 없음'
    if name.upper().split('.')[0] in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
        name='_'+name
    return name+'.pptx'


def create_pptx(documents, slide_size, target):
    # Each visible layer becomes a picture object so it remains independently
    # selectable in PowerPoint. The canvas bounds clip off-canvas layer pixels.
    import io
    import tempfile
    try:
        from pptx import Presentation
        from pptx.util import Inches
    except ModuleNotFoundError as error:
        if error.name != 'pptx': raise
        # The editor may have been started with a Python installation that
        # does not include its user site. Reuse the project's installed
        # dependencies instead of requiring a different launcher.
        import importlib
        project_site=os.path.join(os.path.dirname(os.path.abspath(__file__)),'.venv','Lib','site-packages')
        if os.path.isdir(project_site) and project_site not in sys.path:
            sys.path.append(project_site)
        importlib.invalidate_caches()
        try:
            from pptx import Presentation
            from pptx.util import Inches
        except ModuleNotFoundError as missing:
            if missing.name=='pptx':
                raise ModuleNotFoundError('PPT 내보내기 구성요소가 없습니다. 프로그램 폴더의 실행.bat으로 실행해 주세요.') from missing
            raise
    prs=Presentation()
    prs.slide_width=Inches(40/3 if slide_size=='wide' else 10)
    prs.slide_height=Inches(7.5)
    sw,sh=int(prs.slide_width),int(prs.slide_height)
    for doc in documents:
        cw,ch=doc['canvas_size']
        scale=min(sw/cw,sh/ch)
        offset_x=(sw-cw*scale)/2;offset_y=(sh-ch*scale)/2
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        result_mask=doc.get('result')
        full_mask=result_mask.resize((cw,ch),R.LANCZOS) if result_mask is not None else None
        for layer in doc['layers']:
            if not layer['visible']: continue
            src=layer['image'];x,y=int(layer['x']),int(layer['y'])
            left,top=max(0,-x),max(0,-y)
            right,bottom=min(src.width,cw-x),min(src.height,ch-y)
            if right<=left or bottom<=top:continue
            cropped=src.crop((left,top,right,bottom)).convert('RGBA')
            if full_mask is not None:
                local=full_mask.crop((x+left,y+top,x+right,y+bottom))
                cropped.putalpha(ImageChops.multiply(cropped.getchannel('A'),local))
            if cropped.getchannel('A').getbbox() is None: continue
            data=io.BytesIO();cropped.save(data,format='PNG');data.seek(0)
            shape=slide.shapes.add_picture(data,round(offset_x+(x+left)*scale),round(offset_y+(y+top)*scale),width=round((right-left)*scale),height=round((bottom-top)*scale))
            shape.name=layer['name']
    directory=os.path.dirname(os.path.abspath(target))
    fd,temp_path=tempfile.mkstemp(prefix='.mini-ppt-',suffix='.pptx',dir=directory)
    os.close(fd)
    try:
        prs.save(temp_path)
        os.replace(temp_path,target)
    finally:
        if os.path.exists(temp_path): os.unlink(temp_path)


class PptExportWorker(QThread):
    done=pyqtSignal(str)
    failed=pyqtSignal(str)
    def __init__(self, documents, slide_size, target):
        super().__init__();self.documents=documents;self.slide_size=slide_size;self.target=target
    def run(self):
        try:
            create_pptx(self.documents,self.slide_size,self.target)
            self.done.emit(self.target)
        except Exception as e:
            import traceback
            try:
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'오류기록.log'),'a',encoding='utf-8') as log:
                    log.write('\n[PPT 내보내기 오류]\n'+traceback.format_exc())
            except OSError:pass
            self.failed.emit(f'{type(e).__name__}: {e}')


class NewDocumentDialog(QDialog):
    # ISO 216 paper sizes in millimeters; screen sizes are fixed pixels.
    PRESETS = {'직접 입력': None, 'A4 · 210 × 297 mm': ('paper', 210, 297),
               'A3 · 297 × 420 mm': ('paper', 297, 420),
               'B4 (ISO) · 250 × 353 mm': ('paper', 250, 353),
               'B3 (ISO) · 353 × 500 mm': ('paper', 353, 500),
               'PPT 와이드 16:9 · 1920 × 1080': ('pixels', 1920, 1080),
               'PPT 표준 4:3 · 1200 × 900': ('pixels', 1200, 900),
               'FHD · 1920 × 1080': ('pixels', 1920, 1080),
               'QHD · 2560 × 1440': ('pixels', 2560, 1440),
               '4K UHD · 3840 × 2160': ('pixels', 3840, 2160)}
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle('새 문서'); self.setMinimumWidth(400)
        self.updating = False
        self.setStyleSheet('''
            QDialog { background: #1d2330; color: #eef2fa; }
            QLabel { background: transparent; color: #e7edf7; }
            QComboBox, QSpinBox, QLineEdit { background: #2b3545; color: #f5f7fb; border: 1px solid #536075;
                border-radius: 5px; padding: 7px; min-height: 20px; selection-background-color: #526ff0; }
            QComboBox QAbstractItemView { background: #2b3545; color: #f5f7fb; selection-background-color: #526ff0; }
            QPushButton { background: #354155; color: #f5f7fb; border: 1px solid #536075; border-radius: 6px; padding: 9px 18px; }
            QPushButton:hover { background: #44536b; }
            QPushButton#makeDocument { background: #526ff0; border: none; font-weight: 700; }
            QPushButton#makeDocument:hover { background: #657ff7; }
        ''')
        layout = QVBoxLayout(self); layout.setContentsMargins(24,22,24,22); layout.setSpacing(12)
        title=QLabel('새 문서'); title.setStyleSheet('font-size: 20px; font-weight: bold;'); layout.addWidget(title)
        layout.addWidget(QLabel('이름'))
        self.name_input=QLineEdit('제목 없음'); layout.addWidget(self.name_input)
        layout.addWidget(QLabel('프리셋'))
        self.preset=QComboBox(); self.preset.addItems(self.PRESETS.keys()); layout.addWidget(self.preset)
        row=QHBoxLayout()
        for label, attr, default in [('너비', 'width_input', 1920), ('높이', 'height_input', 1080)]:
            column=QVBoxLayout(); column.addWidget(QLabel(label))
            field=QSpinBox(); field.setRange(1,12000); field.setButtonSymbols(QSpinBox.NoButtons); field.setSuffix(' px'); field.setValue(default)
            setattr(self,attr,field); column.addWidget(field); row.addLayout(column)
        layout.addLayout(row)
        row=QHBoxLayout()
        column=QVBoxLayout();column.addWidget(QLabel('해상도'))
        self.dpi=QSpinBox();self.dpi.setRange(36,600);self.dpi.setValue(75);self.dpi.setSuffix(' dpi');self.dpi.setButtonSymbols(QSpinBox.NoButtons);column.addWidget(self.dpi);row.addLayout(column)
        column=QVBoxLayout();column.addWidget(QLabel('방향'))
        self.orientation=QComboBox();self.orientation.addItems(['기본 방향','가로','세로']);column.addWidget(self.orientation);row.addLayout(column)
        layout.addLayout(row)
        layout.addWidget(QLabel('배경'))
        self.background=QComboBox();self.background.addItems(['투명','흰색']);layout.addWidget(self.background)
        self.hint=QLabel('용지 프리셋은 ISO 규격이며, 해상도에 따라 픽셀 크기가 계산됩니다.');self.hint.setWordWrap(True);self.hint.setStyleSheet('color: #aebdd1; font-size: 11px;');layout.addWidget(self.hint)
        buttons=QHBoxLayout();buttons.addStretch();cancel=QPushButton('취소');cancel.clicked.connect(self.reject);buttons.addWidget(cancel)
        create=QPushButton('만들기');create.setObjectName('makeDocument');create.clicked.connect(self.validate_and_accept);buttons.addWidget(create);layout.addLayout(buttons)
        self.preset.currentIndexChanged.connect(self.apply_preset)
        self.dpi.valueChanged.connect(self.apply_preset)
        self.orientation.currentIndexChanged.connect(self.apply_preset)
        self.width_input.valueChanged.connect(self.mark_custom)
        self.height_input.valueChanged.connect(self.mark_custom)
    def mark_custom(self):
        if self.updating: return
        self.updating=True; self.preset.setCurrentIndex(0); self.updating=False
    def apply_preset(self, *args):
        if self.updating: return
        preset=self.PRESETS[self.preset.itemData(self.preset.currentIndex(),Qt.UserRole+21) or self.preset.currentText()]
        if preset is None:
            width,height=self.width_input.value(),self.height_input.value()
        else:
            kind,width,height=preset
            if kind=='paper':
                width=round(width*self.dpi.value()/25.4);height=round(height*self.dpi.value()/25.4)
        if self.orientation.currentIndex()==1 and width<height: width,height=height,width
        elif self.orientation.currentIndex()==2 and width>height: width,height=height,width
        self.updating=True;self.width_input.setValue(width);self.height_input.setValue(height);self.updating=False
    def validate_and_accept(self):
        if self.width_input.value()*self.height_input.value()>50_000_000:
            QMessageBox.warning(self,'크기가 너무 큽니다','저사양 환경을 위해 최대 5,000만 픽셀까지 만들 수 있습니다. 크기나 해상도를 낮춰 주세요.')
            return
        self.accept()
    def settings(self):
        return (self.name_input.text().strip() or '제목 없음', self.width_input.value(), self.height_input.value(),
                self.dpi.value(), self.background.itemData(self.background.currentIndex(),Qt.UserRole+21) or self.background.currentText())


def scrub_cursor():
    pix = QPixmap(40, 32); pix.fill(Qt.transparent)
    painter = QPainter(pix); painter.setRenderHint(QPainter.Antialiasing)
    hand = QPainterPath(); hand.moveTo(15, 25); hand.lineTo(12, 19); hand.cubicTo(11, 17, 13, 16, 14, 17)
    hand.lineTo(16, 19); hand.lineTo(16, 7); hand.cubicTo(16, 4, 19, 4, 19, 7)
    hand.lineTo(19, 14); hand.cubicTo(19, 11, 22, 11, 22, 14)
    hand.cubicTo(22, 12, 25, 12, 25, 15); hand.cubicTo(25, 13, 28, 14, 28, 17)
    hand.lineTo(27, 23); hand.lineTo(25, 26); hand.closeSubpath()
    painter.setPen(QPen(QColor('#202733'), 1.5)); painter.setBrush(QColor('#f6f8fc')); painter.drawPath(hand)
    painter.setPen(QPen(QColor('#202733'), 3)); painter.drawLine(3, 16, 11, 16); painter.drawLine(29, 16, 37, 16)
    painter.setPen(QPen(QColor('#f6f8fc'), 1)); painter.drawLine(3, 16, 11, 16); painter.drawLine(29, 16, 37, 16)
    painter.setBrush(QColor('#f6f8fc')); painter.setPen(QPen(QColor('#202733'), 1))
    for points in (((2,16),(7,12),(7,20)),((38,16),(33,12),(33,20))):
        path=QPainterPath();path.moveTo(*points[0]);path.lineTo(*points[1]);path.lineTo(*points[2]);path.closeSubpath();painter.drawPath(path)
    painter.end(); return QCursor(pix, 18, 7)


class DocumentNameEdit(QLineEdit):
    renamed = pyqtSignal(str)
    def __init__(self, text):
        super().__init__(text); self.setReadOnly(True); self.setFocusPolicy(Qt.ClickFocus)
        self.setCursor(Qt.PointingHandCursor); self.setMaxLength(200); self.setFixedWidth(190)
        self.setToolTip('클릭해서 현재 탭 이름 변경')
        self._original_name = text; self._editing_name = False
        self.editingFinished.connect(self.finish_edit)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._editing_name:
            window = self.window()
            if getattr(window, 'active_document', -1) < 0 or getattr(window.canvas, 'busy', False): return
            self._original_name = self.text(); self._editing_name = True; self.setReadOnly(False)
            self.setCursor(Qt.IBeamCursor); self.setFocus(); self.selectAll(); event.accept(); return
        super().mousePressEvent(event)
    def finish_edit(self):
        if not self._editing_name: return
        name = self.text().strip() or self._original_name
        self._editing_name = False; self.setReadOnly(True); self.setCursor(Qt.PointingHandCursor)
        self.setText(name)
        if name != self._original_name: self.renamed.emit(name)
    def event(self, event):
        if getattr(self, '_editing_name', False) and event.type() == QEvent.ShortcutOverride:
            event.accept(); return True
        return super().event(event)
    def keyPressEvent(self, event):
        if self._editing_name and event.key() == Qt.Key_Escape:
            self.setText(self._original_name); self.finish_edit(); self.clearFocus(); event.accept(); return
        if self._editing_name and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.finish_edit(); self.clearFocus(); event.accept(); return
        super().keyPressEvent(event)


class ScrubLabel(QLabel):
    def __init__(self, text, target=None):
        super().__init__(text); self.target=target; self.drag_origin=None
        self.setCursor(scrub_cursor()); self.setToolTip('좌우로 드래그해 값을 조절합니다.')
    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton and self.target is not None and self.target.isEnabled():
            self.drag_origin=event.globalPos().x();self.start_value=self.target.value();event.accept();return
        super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        if self.drag_origin is not None:
            delta=event.globalPos().x()-self.drag_origin
            self.target.setValue(self.start_value + round(delta/3)*self.target.singleStep());event.accept();return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        if event.button()==Qt.LeftButton and self.drag_origin is not None:
            delta=event.globalPos().x()-self.drag_origin
            self.target.setValue(self.start_value + round(delta/3)*self.target.singleStep());self.drag_origin=None;event.accept();return
        super().mouseReleaseEvent(event)


class DimensionSpinBox(QSpinBox):
    def focusInEvent(self, event):
        super().focusInEvent(event)
        # Selecting just the numeric portion makes replacement possible with
        # one click, while leaving the pixel suffix managed by QSpinBox.
        QTimer.singleShot(0, self.selectAll)


class ResizeDialog(QDialog):
    def __init__(self, title, width, height, parent=None, position=None, aspect=True):
        super().__init__(parent); self.setWindowFlag(Qt.WindowContextHelpButtonHint, False); self.setWindowTitle(title); self.setMinimumWidth(350)
        self.setStyleSheet('''QDialog { background:#1d2330; color:#eef2fa; } QLabel,QCheckBox { background:transparent; color:#e7edf7; }
            QSpinBox { background:#2b3545; color:white; border:1px solid #536075; border-radius:5px; padding:7px; selection-background-color:#526ff0; }
            QSpinBox:focus { border: 1px solid #8296ff; }
            QPushButton { background:#354155; color:white; border:1px solid #536075; border-radius:5px; padding:8px 16px; }
            QPushButton#applySize { background:#526ff0; border:none; }''')
        self.ratio = width / height; self.updating = False
        layout=QVBoxLayout(self);layout.setContentsMargins(22,20,22,20);layout.setSpacing(10)
        title_label=QLabel(title);title_label.setStyleSheet('font-size:18px; font-weight:bold;');layout.addWidget(title_label)
        for label,attr,value,limits in [('너비','width_input',width,(1,12000)),('높이','height_input',height,(1,12000))]:
            row=QHBoxLayout(); box=DimensionSpinBox();row.addWidget(ScrubLabel(label,box));box.setRange(*limits);box.setValue(value);box.setSuffix(' px');box.setButtonSymbols(QSpinBox.NoButtons);box.setKeyboardTracking(True);row.addWidget(box,1);setattr(self,attr,box);layout.addLayout(row)
        self.lock=QCheckBox('가로세로 비율 유지');self.lock.setChecked(aspect);self.lock.setVisible(aspect);layout.addWidget(self.lock)
        self.x_input=self.y_input=None
        if position is not None:
            for label,attr,value in [('가로 위치','x_input',position[0]),('세로 위치','y_input',position[1])]:
                row=QHBoxLayout();row.addWidget(QLabel(label));box=QSpinBox();box.setRange(-50000,50000);box.setValue(value);box.setSuffix(' px');box.setButtonSymbols(QSpinBox.NoButtons);row.addWidget(box,1);setattr(self,attr,box);layout.addLayout(row)
        if position is None and not aspect:
            note=QLabel('기존 내용은 중앙에 유지됩니다.');note.setWordWrap(True);layout.addWidget(note)
        elif position is None and aspect:
            note=QLabel('캔버스와 모든 레이어의 크기를 함께 변경합니다. 개별 레이어는 Ctrl+T로 변형하세요.');note.setWordWrap(True);note.setStyleSheet('color:#b5c2d4; font-size:12px;');layout.addWidget(note)
        buttons=QHBoxLayout();buttons.addStretch();cancel=QPushButton('취소');cancel.clicked.connect(self.reject);buttons.addWidget(cancel);apply=QPushButton('적용');apply.setObjectName('applySize');apply.setDefault(True);apply.clicked.connect(self.apply);buttons.addWidget(apply);layout.addLayout(buttons)
        self.width_input.valueChanged.connect(self.width_changed);self.height_input.valueChanged.connect(self.height_changed)
    def width_changed(self,value):
        if not self.updating and self.lock.isChecked():
            self.updating=True;self.height_input.setValue(max(1,round(value/self.ratio)));self.updating=False
    def height_changed(self,value):
        if not self.updating and self.lock.isChecked():
            self.updating=True;self.width_input.setValue(max(1,round(value*self.ratio)));self.updating=False
    def apply(self):
        self.width_input.interpretText(); self.height_input.interpretText()
        if self.width_input.value()*self.height_input.value()>50_000_000:
            QMessageBox.warning(self,'크기가 너무 큽니다','최대 5,000만 픽셀까지 설정할 수 있습니다.');return
        self.accept()
    def values(self):
        return (self.width_input.value(),self.height_input.value(),self.x_input.value() if self.x_input else None,self.y_input.value() if self.y_input else None)


class SelectionRadiusDialog(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent);self.setWindowFlag(Qt.WindowContextHelpButtonHint,False)
        self.setWindowTitle(title);self.setMinimumWidth(340)
        self.setStyleSheet("""QDialog { background:#1d2330; color:#eef2fa; }
            QLabel { background:transparent; color:#e7edf7; }
            QFrame#radiusControl { background:#252c38; border:1px solid #536075; border-radius:7px; }
            QSpinBox { background:transparent; color:#eff4fc; border:0; padding:5px; font-weight:600; selection-background-color:#526ff0; }
            QPushButton { background:#303949; color:#f4f7fc; border:1px solid #536075; border-radius:5px; padding:7px 14px; }
            QPushButton:hover { background:#3b485d; }
            QPushButton#radiusStep { background:transparent; border:0; padding:0; font-size:17px; }
            QPushButton#radiusStep:hover { background:#44536a; }
            QPushButton#primary { background:#526ff0; color:white; border:0; }""")
        layout=QVBoxLayout(self);layout.setContentsMargins(20,18,20,18);layout.setSpacing(10)
        heading=QLabel(title);heading.setStyleSheet('font-size:17px; font-weight:bold;');layout.addWidget(heading)
        layout.addWidget(QLabel('반경 (원본 픽셀)'))
        control=QFrame();control.setObjectName('radiusControl');row=QHBoxLayout(control);row.setContentsMargins(4,4,4,4);row.setSpacing(3)
        minus=AnimatedButton('−');minus.setObjectName('radiusStep');minus.setFixedSize(34,30);minus.setToolTip('반경 줄이기');row.addWidget(minus)
        self.radius=DimensionSpinBox();self.radius.setRange(1,500);self.radius.setValue(3);self.radius.setSuffix(' px');self.radius.setAlignment(Qt.AlignCenter);self.radius.setButtonSymbols(QSpinBox.NoButtons);self.radius.setKeyboardTracking(False);row.addWidget(self.radius,1)
        plus=AnimatedButton('+');plus.setObjectName('radiusStep');plus.setFixedSize(34,30);plus.setToolTip('반경 늘리기');row.addWidget(plus)
        minus.clicked.connect(self.radius.stepDown);plus.clicked.connect(self.radius.stepUp);layout.addWidget(control)
        buttons=QHBoxLayout();buttons.addStretch();cancel=QPushButton('취소');cancel.clicked.connect(self.reject);buttons.addWidget(cancel)
        apply=QPushButton('적용');apply.setObjectName('primary');apply.setDefault(True);apply.clicked.connect(self.apply);buttons.addWidget(apply);layout.addLayout(buttons)
        self.radius.setFocus();QTimer.singleShot(0,self.radius.selectAll)
    def apply(self):
        self.radius.interpretText();self.accept()


class ExportDialog(QDialog):
    def __init__(self, image, settings, parent=None, settings_only=False, layer_count=1):
        super().__init__(parent);self.setWindowFlag(Qt.WindowContextHelpButtonHint,False)
        self.layer_count=layer_count
        self.setWindowTitle('내보내기 기본 설정' if settings_only else '내보내기 형식');self.resize(800,520)
        self.setStyleSheet("""QDialog { background:#1d2330; color:#eef2fa; } QLabel,QCheckBox { color:#e7edf7; background:transparent; }
            QComboBox,QSpinBox,QDoubleSpinBox { background:#293442; color:#eff4fc; border:1px solid #536075; border-radius:5px; padding:7px; }
            QPushButton { background:#303949; color:#f4f7fc; border:1px solid #536075; border-radius:5px; padding:8px 16px; }
            QPushButton:hover { background:#3b485d; } QPushButton#primary { background:#526ff0; color:white; }
            QLabel#exportPreview { background:#20242e; border:1px solid #465266; border-radius:5px; }""")
        root=QVBoxLayout(self);root.setContentsMargins(18,18,18,18);row=QHBoxLayout();root.addLayout(row,1)
        preview=QLabel();preview.setObjectName('exportPreview');preview.setAlignment(Qt.AlignCenter);preview.setMinimumSize(300,280)
        preview.setPixmap(preview_pixmap(image).scaled(480,380,Qt.KeepAspectRatio,Qt.SmoothTransformation));row.addWidget(preview,1)
        panel=QVBoxLayout();panel.setSpacing(12);row.addLayout(panel);panel.addWidget(QLabel('파일 설정'))
        panel.addWidget(QLabel('형식'));self.format=QComboBox();self.format.addItem('PNG','PNG');self.format.addItem('JPG','JPEG');self.format.addItem('GIF','GIF');self.format.setCurrentIndex(max(0,self.format.findData(settings.get('format','PNG'))));panel.addWidget(self.format)
        self.transparency=QCheckBox('투명 배경 유지');self.transparency.setChecked(settings.get('transparency',True));panel.addWidget(self.transparency)
        self.quality_label=ScrubLabel('JPG 품질');panel.addWidget(self.quality_label);self.quality=QSpinBox();self.quality_label.target=self.quality;self.quality.setRange(1,100);self.quality.setValue(settings.get('quality',90));self.quality.setSuffix(' %');self.quality.setButtonSymbols(QSpinBox.NoButtons);panel.addWidget(self.quality)
        self.animation=QCheckBox('레이어별 애니메이션');self.animation.setChecked(settings.get('animate',True));panel.addWidget(self.animation)
        self.frame_label=ScrubLabel('프레임 간격');panel.addWidget(self.frame_label)
        self.frame_ms=QDoubleSpinBox();self.frame_label.target=self.frame_ms;self.frame_ms.setDecimals(2);self.frame_ms.setRange(0.02,60);self.frame_ms.setSingleStep(0.1);self.frame_ms.setValue(settings.get('frame_ms',2000)/1000);self.frame_ms.setSuffix(' s');self.frame_ms.setButtonSymbols(QDoubleSpinBox.NoButtons);panel.addWidget(self.frame_ms)
        self.repeat=QCheckBox('반복 재생');self.repeat.setChecked(settings.get('repeat',True));panel.addWidget(self.repeat)
        self.frame_note=QLabel('아래쪽 레이어부터 순서대로 하나씩 쌓입니다.' if layer_count>=2 else '애니메이션 GIF를 만들려면 보이는 레이어가 두 개 이상 필요합니다.');self.frame_note.setWordWrap(True);panel.addWidget(self.frame_note)
        self.note=QLabel('JPG는 투명 배경이 흰색으로 저장됩니다. GIF는 투명·불투명만 지원합니다.');self.note.setWordWrap(True);panel.addWidget(self.note)
        info=QLabel(f'{image.width} × {image.height} px');panel.addWidget(info);panel.addStretch();self.format.currentIndexChanged.connect(self.update_format);self.animation.toggled.connect(self.update_format);self.update_format()
        buttons=QHBoxLayout();buttons.addStretch();cancel=QPushButton('취소');cancel.clicked.connect(self.reject);buttons.addWidget(cancel)
        confirm=QPushButton('저장' if settings_only else '내보내기');confirm.setObjectName('primary');confirm.clicked.connect(self.accept);buttons.addWidget(confirm);root.addLayout(buttons)
    def update_format(self):
        fmt=self.format.currentData();self.transparency.setEnabled(fmt!='JPEG');self.quality.setEnabled(fmt=='JPEG');self.quality_label.setEnabled(fmt=='JPEG')
        is_gif=fmt=='GIF';self.animation.setVisible(is_gif);self.animation.setEnabled(is_gif and self.layer_count>=2)
        enabled=is_gif and self.layer_count>=2 and self.animation.isChecked()
        for widget in (self.frame_label,self.frame_ms,self.repeat):widget.setVisible(is_gif);widget.setEnabled(enabled)
        self.frame_note.setVisible(is_gif)
    def values(self):return {'format':self.format.currentData(),'quality':self.quality.value(),'transparency':self.transparency.isChecked(),'animate':self.animation.isChecked() and self.layer_count>=2,'frame_ms':round(self.frame_ms.value()*1000),'repeat':self.repeat.isChecked()}


class SettingsDialog(QDialog):
    def __init__(self, config, parent):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle('설정'); self.resize(640, 550); self.setMinimumWidth(580)
        self.setStyleSheet("""QDialog { background:#1d2330; color:#eef2fa; }
            QLabel,QCheckBox { color:#e7edf7; background:transparent; }
            QSpinBox,QComboBox,QKeySequenceEdit { background:#293442; color:#eff4fc; border:1px solid #536075; border-radius:5px; padding:7px; }
            QPushButton { background:#303949; color:#f4f7fc; border:1px solid #536075; border-radius:5px; padding:8px 13px; }
            QPushButton:hover { background:#3b485d; } QPushButton#primary { background:#526ff0; color:white; }
            QTabWidget::pane { border:1px solid #536075; border-radius:5px; }
            QTabBar::tab { background:#252e3b; color:#b9c6d7; padding:9px 12px; min-width:0; min-height:26px; border:1px solid #465266; border-top-left-radius:5px; border-top-right-radius:5px; }
            QTabBar::tab:hover:!selected { background:#3b485d; color:#f4f7fc; }
            QTabBar::tab:selected { background:#354156; color:#f4f7fc; border-top:2px solid #8296ff; }
            QScrollArea { border:0; background:transparent; }""")
        root=QVBoxLayout(self); root.setContentsMargins(20,20,20,20)
        title=QLabel('설정'); title.setStyleSheet('font-size:20px; font-weight:bold;'); root.addWidget(title)
        tabs=QTabWidget(); self.tabs=tabs; tabs.tabBar().setExpanding(True);tabs.tabBar().setUsesScrollButtons(False); root.addWidget(tabs,1)
        page=QWidget(); box=QVBoxLayout(page); box.setContentsMargins(18,18,18,18); box.setSpacing(12)
        box.addWidget(QLabel('문서별 실행 취소 기록 메모리 예산'))
        self.history_control=QFrame();self.history_control.setObjectName('historyControl')
        history_row=QHBoxLayout(self.history_control);history_row.setContentsMargins(4,4,4,4);history_row.setSpacing(4)
        self.history_minus=AnimatedButton('−');self.history_minus.setFixedSize(38,30);self.history_minus.setToolTip('메모리 예산 줄이기');history_row.addWidget(self.history_minus)
        self.history_limit=QSpinBox(); self.history_limit.setRange(64,8192); self.history_limit.setSingleStep(64); self.history_limit.setSuffix(' MB'); self.history_limit.setValue(config['history_mb']);self.history_limit.setButtonSymbols(QSpinBox.NoButtons);self.history_limit.setAlignment(Qt.AlignCenter);history_row.addWidget(self.history_limit,1)
        self.history_plus=AnimatedButton('+');self.history_plus.setFixedSize(38,30);self.history_plus.setToolTip('메모리 예산 늘리기');history_row.addWidget(self.history_plus)
        self.history_minus.clicked.connect(self.history_limit.stepDown);self.history_plus.clicked.connect(self.history_limit.stepUp);box.addWidget(self.history_control)
        help_text=QLabel('기록의 예상 메모리를 문서별로 제한합니다. 값을 줄이면 오래된 실행 취소 기록이 삭제될 수 있습니다. 현재 이미지와 AI 모델을 포함한 프로그램 전체 RAM 한도는 아닙니다.')
        help_text.setWordWrap(True); box.addWidget(help_text)
        self.memory_saving=QCheckBox('AI 인식 메모리 절약 사용'); self.memory_saving.setChecked(config['memory_saving']); box.addWidget(self.memory_saving)
        tip=QLabel('AI 메모리 설정은 다음 인물 자동 선택부터 적용됩니다.');tip.setWordWrap(True);box.addWidget(tip)
        clear=QPushButton('임시 캐시 비우기');box.addWidget(clear)
        self.cache_status=QLabel('사진과 실행 취소 기록은 유지됩니다.');self.cache_status.setWordWrap(True);box.addWidget(self.cache_status)
        clear.clicked.connect(self.clear_cache);box.addStretch();tabs.addTab(page,'캐시 · 메모리')
        page=QWidget();box=QVBoxLayout(page);box.setContentsMargins(18,18,18,18);box.addWidget(QLabel('화면 테마'))
        self.theme=QComboBox();self.theme.setView(QListView());self.theme.addItem('다크 모드','dark');self.theme.addItem('라이트 모드','light');self.theme.setCurrentIndex(self.theme.findData(config['theme']));box.addWidget(self.theme);box.addStretch();tabs.addTab(page,'화면')
        page=QWidget();box=QVBoxLayout(page);box.setContentsMargins(18,18,18,18)
        box.addWidget(QLabel('프로그램 언어'))
        self.language=QComboBox();self.language.setView(QListView());self.language.addItem('한국어','ko');self.language.addItem('English','en');self.language.setCurrentIndex(self.language.findData(config.get('language','ko')));box.addWidget(self.language)
        language_note=QLabel('언어 변경 후 저장을 누르세요.');language_note.setWordWrap(True);box.addWidget(language_note);box.addStretch();tabs.addTab(page,'언어')
        page=QWidget();box=QVBoxLayout(page);box.setContentsMargins(18,18,18,18)
        info=QLabel('변경할 항목을 클릭한 후 원하는 키 조합을 누르세요.');info.setWordWrap(True);box.addWidget(info)
        area=QScrollArea();area.setWidgetResizable(True);content=QWidget();rows=QVBoxLayout(content);rows.setContentsMargins(0,0,0,0)
        self.key_edits={}
        for name, label in SHORTCUT_LABELS.items():
            row=QHBoxLayout();row.addWidget(QLabel(label),1);edit=QKeySequenceEdit();edit.setKeySequence(QKeySequence(config['shortcuts'][name]));row.addWidget(edit,1);rows.addLayout(row);self.key_edits[name]=edit
        rows.addStretch();area.setWidget(content);box.addWidget(area,1)
        reset=QPushButton('단축키 기본값 복원');reset.clicked.connect(self.reset_shortcuts);box.addWidget(reset);tabs.addTab(page,'단축키')
        buttons=QHBoxLayout();buttons.addStretch();cancel=QPushButton('취소');cancel.clicked.connect(self.reject);buttons.addWidget(cancel)
        save=QPushButton('저장');save.setObjectName('primary');save.clicked.connect(self.save_values);buttons.addWidget(save);root.addLayout(buttons)
        self.values=None
        self._tab_animations=[]
        for i in range(tabs.count()):
            effect=QGraphicsOpacityEffect(tabs.widget(i));effect.setOpacity(1.0);tabs.widget(i).setGraphicsEffect(effect)
            animation=QPropertyAnimation(effect,b'opacity',self);animation.setDuration(180);animation.setEasingCurve(QEasingCurve.OutCubic);self._tab_animations.append(animation)
        tabs.currentChanged.connect(self.animate_tab)
    def showEvent(self, event):
        super().showEvent(event)
        if sys.platform == 'win32':
            import ctypes
            hwnd=int(self.winId());user32=ctypes.windll.user32
            user32.SetWindowLongW(hwnd,-20,user32.GetWindowLongW(hwnd,-20)|0x00000001)
            user32.SetWindowPos(hwnd,0,0,0,0,0,0x0001|0x0002|0x0004|0x0020)
    def animate_tab(self,index):
        if 0<=index<len(self._tab_animations):
            animation=self._tab_animations[index];animation.stop();animation.setStartValue(0.45);animation.setEndValue(1.0);animation.start()
    def clear_cache(self):
        self.parent().clear_temporary_cache();self.cache_status.setText('임시 캐시를 비웠습니다. 사진과 실행 취소 기록은 유지됩니다.')
    def reset_shortcuts(self):
        for name, edit in self.key_edits.items(): edit.setKeySequence(QKeySequence(SHORTCUT_DEFAULTS[name]))
    def save_values(self):
        keys={}; reserved={QKeySequence(item).toString(QKeySequence.PortableText) for item in FIXED_SHORTCUTS}
        for name, edit in self.key_edits.items():
            seq=edit.keySequence(); value=seq.toString(QKeySequence.PortableText)
            if not value or seq.count()!=1 or value in reserved or value in keys.values():
                QMessageBox.warning(self,'단축키 확인','비어 있거나 중복된 단축키, 또는 다른 기능에서 사용하는 단축키입니다. 각 항목에 서로 다른 키 조합 하나를 지정해 주세요.');return
            keys[name]=value
        self.values={'theme':self.theme.currentData(),'language':self.language.currentData(),'history_mb':self.history_limit.value(),'memory_saving':self.memory_saving.isChecked(),'shortcuts':keys,'export':dict(self.parent().preferences.get('export',{'format':'PNG','quality':90,'transparency':True}))};self.accept()


class Window(QMainWindow):
    windows = []
    def __init__(self):
        super().__init__()
        self.preferences = load_settings()
        global _RUNTIME_MEMORY_SAVING
        _RUNTIME_MEMORY_SAVING = self.preferences['memory_saving']
        self.setAcceptDrops(True)
        self.setWindowTitle('미니 포토샵')
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)),'assets','mini_photoshop.ico')))
        self.resize(1440, 900)
        self.original = None
        self.zoom_percent = 100
        self.zoom_anchor = None
        self.tolerances = {'subject': 30, 'brush': 30, 'wand': 30, 'object': 30}
        self.tolerance_key = 'subject'
        self.worker = None
        self._local_clipboard_image = None
        self._local_clipboard_name = None
        self._local_clipboard_sequence = None
        self.documents = []
        self.active_document = -1
        self.switching_document = False
        self.setStyleSheet('''
            QWidget { background: #151922; color: #e9edf5; font-family: "Malgun Gothic"; font-size: 13px; }
            QLabel { background: transparent; color: #e8edf5; }
            QToolTip { color: #f4f7fc; background-color: #2a3342; border: 1px solid #566780;
                       border-radius: 5px; padding: 7px 10px; font-size: 12px; }
            QFrame#toolbar { background: #303030; border-bottom: 1px solid #111; }
            QFrame#toolbar QPushButton { padding: 4px 9px; border-radius: 3px; }
            QLineEdit#fileName { background: transparent; color: #e7eaf0; border: 1px solid transparent; border-radius: 4px; padding: 4px 6px; font-weight: 600; }
            QLineEdit#fileName:!read-only { background: #293442; border: 1px solid #8296ff; }
            QTabBar { background: #191f29; border-bottom: 1px solid #3a4657; }
            QTabBar::tab { background: #252e3b; color: #b9c6d7; border: 1px solid #3b485a;
                           border-bottom: none; border-top-left-radius: 5px; border-top-right-radius: 5px;
                           min-width: 100px; min-height: 27px; padding: 2px 8px; margin-right: 2px; }
            QTabBar::tab:selected { background: #354156; color: #f4f7fc; border-top: 2px solid #8296ff; }
            QTabBar::tab:hover:!selected { background: #303c4e; color: #f4f7fc; }
            QFrame#tabRow { background: #191f29; border-bottom: 1px solid #3a4657; }
            QFrame#emptyTab { background: #354156; border: 1px solid #3b485a; border-top: 2px solid #8296ff; border-bottom: none; border-top-left-radius: 5px; border-top-right-radius: 5px; }
            QPushButton#newTabButton { background: transparent; color: #c2ccdb; border: none; border-radius: 5px; padding: 0; font-size: 23px; }
            QPushButton#newTabButton:hover { background: #3b485d; color: #f4f7fc; }
            QTabBar QPushButton#tabClose { background: transparent; border: none; border-radius: 4px; padding: 0; }
            QTabBar QPushButton#tabClose:hover { background: #56647a; }
            QTabBar QPushButton#tabClose:pressed { background: #7184a0; }
            QLabel#brand { font-size: 15px; font-weight: 700; color: white; }
            QFrame#side { background: #1d2330; border-right: 1px solid #303849; }
            QLabel#title { font-size: 22px; font-weight: 700; color: white; }
            QLabel#section { color: #d1dae8; font-size: 12px; font-weight: 700; }
            QLabel#hint { color: #c2ccdb; font-size: 13px; }
            QLabel#status { color: #aab7c9; padding: 10px 15px; background: #1b202a; }
            QPushButton { background: #303949; color: #f4f7fc; border: 1px solid #414d60; border-radius: 7px; padding: 10px; text-align: center; }
            QPushButton:hover { background: #3b485d; }
            QPushButton:disabled { color: #707b8b; background: #252c38; }
            QPushButton#primary { background: #526ff0; border: none; font-weight: 700; }
            QPushButton#primary:hover { background: #6782ff; }
            QPushButton:checked { background: #345946; border: 1px solid #54ce8c; }
            QPushButton#remove:checked { background: #61383f; border: 1px solid #f16d7c; }
            QSplitter::handle:horizontal { background: #303849; }
            QSplitter::handle:horizontal:hover { background: #7188f7; }
            QSlider { background: transparent; }
            QSlider::groove:horizontal { height: 5px; background: #303a49; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #7188f7; border-radius: 2px; }
            QSlider::handle:horizontal { background: #a3b2ff; border: 1px solid #c5cfff; width: 14px; margin: -6px 0; border-radius: 7px; }
            QSlider::handle:horizontal:hover { background: #c2ccff; }
            QScrollArea { border: none; background: #20242e; }
            QAbstractScrollArea::corner { background: #202630; }
            QScrollBar:vertical { background: #202630; width: 14px; margin: 0; border: none; }
            QScrollBar::handle:vertical { background: #56657a; min-height: 28px; border: 3px solid #202630; border-radius: 6px; }
            QScrollBar::handle:vertical:hover { background: #798da8; }
            QScrollBar::handle:vertical:pressed { background: #8195ff; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; border: none; background: none; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: #202630; }
            QScrollBar:horizontal { background: #202630; height: 14px; margin: 0; border: none; }
            QScrollBar::handle:horizontal { background: #56657a; min-width: 28px; border: 3px solid #202630; border-radius: 6px; }
            QScrollBar::handle:horizontal:hover { background: #798da8; }
            QScrollBar::handle:horizontal:pressed { background: #8195ff; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; border: none; background: none; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: #202630; }
        ''')
        host = QWidget(); self.setCentralWidget(host)
        outer = QVBoxLayout(host); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        toolbar = QFrame(); toolbar.setObjectName('toolbar')
        toolbar_layout = QVBoxLayout(toolbar); toolbar_layout.setContentsMargins(12, 7, 12, 7); toolbar_layout.setSpacing(5)
        top = QHBoxLayout(); top.setSpacing(7); toolbar_layout.addLayout(top)
        self.save_btn = self.icon_button('save', 'PNG 저장', self.save); top.addWidget(self.save_btn)
        self.undo_btn = self.icon_button('undo', '실행 취소 (Ctrl+Z)', self.undo); top.addWidget(self.undo_btn)
        self.redo_btn = self.icon_button('redo', '다시 실행 (Ctrl+Y)', self.redo); top.addWidget(self.redo_btn)
        self.file_name = DocumentNameEdit('제목 없음'); self.file_name.setObjectName('fileName'); self.file_name.renamed.connect(self.rename_current_tab); top.addSpacing(18); top.addWidget(self.file_name)
        top.addStretch()
        mode_row = QHBoxLayout(); mode_row.setSpacing(7)
        self.file_btn = self.button('파일', lambda: None)
        self.file_menu = QMenu(self)
        self.file_menu.setStyleSheet('''
            QMenu { background: #252e3b; color: #edf2fa; border: 1px solid #536075; padding: 5px; }
            QMenu::item { background: transparent; padding: 8px 36px 8px 14px; border-radius: 4px; }
            QMenu::item:selected { background: #40516b; color: #ffffff; }
            QMenu::item:disabled { color: #7b8798; }
            QMenu::separator { height: 1px; background: #4a576a; margin: 5px 8px; }
        ''')
        actions = [('새로 만들기...', 'Ctrl+N', self.new_document), ('새 탭', 'Ctrl+Shift+N', self.new_tab),
                   ('사진 불러오기...', 'Ctrl+O', self.open), ('사진 붙여넣기', None, self.paste_image),
                   None, ('저장', 'Ctrl+S', self.save), ('다른 이름으로 저장...', 'Ctrl+Shift+S', self.save_as),
                   ('__export__', None, None),
                   None, ('설정...', None, self.open_settings), None, ('탭 닫기', 'Ctrl+W', self.close_current_tab), ('종료', 'Ctrl+Q', self.close)]
        self.shortcut_actions = {}
        for item in actions:
            if item is None: self.file_menu.addSeparator(); continue
            label, shortcut, callback = item
            if label == '__export__':
                self.export_menu=QMenu('내보내기',self.file_menu);self.export_menu.setStyleSheet(self.file_menu.styleSheet());self.file_menu.addMenu(self.export_menu)
                self.quick_export_action=self.export_menu.addAction('PNG로 빠른 내보내기...',self.quick_export_png)
                export_action=self.export_menu.addAction('내보내기 형식...',self.export_as_dialog);self.export_as_action=export_action;export_action.setShortcut('Ctrl+Alt+Shift+W');self.addAction(export_action)
                self.export_menu.addSeparator();self.export_preferences_action=self.export_menu.addAction('내보내기 기본 설정...',self.export_preferences)
                web_action=self.export_menu.addAction('웹용으로 저장...',self.export_as_dialog);self.web_export_action=web_action;web_action.setShortcut('Ctrl+Alt+Shift+S');self.addAction(web_action)
                self.export_menu.addSeparator();self.ppt_export_action=self.export_menu.addAction('PPT로 내보내기...',self.export_ppt)
                continue
            action = self.file_menu.addAction(label); action.triggered.connect(callback)
            if callback.__name__ == 'paste_image':
                self.paste_menu_action=action;action.setShortcutContext(Qt.WidgetShortcut);action.setShortcut(QKeySequence(self.preferences['shortcuts']['paste']))
            if callback.__name__ in ('new_document', 'open', 'save'): self.shortcut_actions[callback.__name__] = action
            if shortcut:
                action.setShortcut(shortcut); self.addAction(action)
        self.file_btn.setMenu(self.file_menu); mode_row.addWidget(self.file_btn)
        self.image_btn = self.button('이미지', lambda: None)
        self.image_menu = QMenu(self); self.image_menu.setStyleSheet(self.file_menu.styleSheet())
        self.image_menu.addAction('이미지 크기…', self.resize_image_dialog)
        self.image_menu.addAction('캔버스 크기…', self.resize_canvas_dialog)
        self.image_btn.setMenu(self.image_menu); mode_row.addWidget(self.image_btn)
        self.select_btn=self.button('선택',lambda:None);self.select_menu=QMenu(self);self.select_menu.setStyleSheet(self.file_menu.styleSheet())
        self.selection_actions={}
        for label,key,callback in [('모두','Ctrl+A',self.select_all),('선택 해제','Ctrl+D',self.deselect),('다시 선택','Ctrl+Shift+D',self.reselect),('반전','Ctrl+Shift+I',self.invert_selection)]:
            action=self.select_menu.addAction(label);action.triggered.connect(callback);action.setShortcut(key);self.addAction(action);self.selection_actions[label]=action
        self.select_menu.addSeparator();self.modify_menu=QMenu('수정',self.select_menu);self.modify_menu.setStyleSheet(self.file_menu.styleSheet());self.select_menu.addMenu(self.modify_menu)
        for name,kind in [('테두리...','border'),('매끄럽게...','smooth'),('확대...','expand'),('축소...','contract'),('페더...','feather')]:
            action=self.modify_menu.addAction(name);action.triggered.connect(lambda checked=False,k=kind:self.modify_selection_dialog(k))
            if kind=='feather':action.setShortcut('Shift+F6');self.addAction(action)
        self.select_btn.setMenu(self.select_menu);mode_row.addWidget(self.select_btn)
        mode_row.addSpacing(10)
        self.mode_switch = ModeSwitch(); self.mode_switch.toggled.connect(lambda checked: self.change_mode(int(checked))); mode_row.addWidget(self.mode_switch)
        mode_row.addSpacing(8)
        self.tolerance_label = ScrubLabel('인물 자동 허용치'); mode_row.addWidget(self.tolerance_label)
        self.tolerance_control = QFrame(); self.tolerance_control.setObjectName('toleranceControl')
        self.tolerance_control.setStyleSheet("""
            QFrame#toleranceControl { background: #252c38; border: 1px solid #566479; border-radius: 5px; }
            QFrame#toleranceControl:disabled { background: #202630; border-color: #38414e; }
            QSpinBox { background: transparent; color: #f4f7fc; border: none; padding: 0; font-weight: 600; }
            QSpinBox:disabled { color: #778391; }
            QPushButton { background: transparent; border: none; border-radius: 4px; color: #c8d5e7; padding: 0; font-size: 16px; }
            QPushButton:hover { background: #44536a; color: white; }
            QPushButton:pressed { background: #526ff0; }
            QPushButton:disabled { color: #637080; background: transparent; }
        """)
        tolerance_layout = QHBoxLayout(self.tolerance_control); tolerance_layout.setContentsMargins(2,2,2,2); tolerance_layout.setSpacing(0)
        self.tolerance_minus = self.button('−', lambda: self.tolerance_box.stepDown()); self.tolerance_minus.setFixedSize(26,28)
        self.tolerance_minus.setToolTip('허용치 낮추기'); tolerance_layout.addWidget(self.tolerance_minus)
        self.tolerance_box = QSpinBox(); self.tolerance_label.target=self.tolerance_box; self.tolerance_box.setRange(0, 100); self.tolerance_box.setValue(30)
        self.tolerance_box.setButtonSymbols(QSpinBox.NoButtons); self.tolerance_box.setAlignment(Qt.AlignCenter)
        self.tolerance_box.setFixedSize(40,28)
        self.tolerance_box.setToolTip('값이 높을수록 경계를 더 너그럽게 선택합니다. 인물 자동 선택은 사람으로 인식한 영역 주변으로 제한됩니다. 다음 선택부터 적용됩니다.')
        self.tolerance_box.valueChanged.connect(self.set_tolerance); tolerance_layout.addWidget(self.tolerance_box)
        self.tolerance_plus = self.button('+', lambda: self.tolerance_box.stepUp()); self.tolerance_plus.setFixedSize(26,28)
        self.tolerance_plus.setToolTip('허용치 높이기'); tolerance_layout.addWidget(self.tolerance_plus)
        mode_row.addWidget(self.tolerance_control)
        mode_row.addSpacing(12)
        self.quality_label=QLabel('인식 품질');mode_row.addWidget(self.quality_label)
        self.quality_box=QComboBox();self.quality_box.setView(QListView());self.quality_box.addItem('정밀 (느림)','precise');self.quality_box.addItem('빠름 (저사양)','fast')
        self.quality_box.setToolTip('정밀: 두 인물 모델을 비교해 빠진 부분 보완 · 빠름: 메모리와 처리 시간 절약')
        self.quality_box.setMinimumWidth(150);mode_row.addWidget(self.quality_box)
        self.brush_size_label = ScrubLabel('선택 크기'); mode_row.addWidget(self.brush_size_label)
        self.brush_control = QFrame(); self.brush_control.setObjectName('toleranceControl')
        self.brush_control.setStyleSheet(self.tolerance_control.styleSheet())
        brush_layout = QHBoxLayout(self.brush_control); brush_layout.setContentsMargins(2,2,2,2); brush_layout.setSpacing(0)
        self.brush_minus = self.button('−', lambda: self.brush_size.stepDown()); self.brush_minus.setFixedSize(26,28)
        self.brush_minus.setToolTip('선택 크기 줄이기'); brush_layout.addWidget(self.brush_minus)
        self.brush_size = QSpinBox(); self.brush_size_label.target=self.brush_size; self.brush_size.setRange(8, 120); self.brush_size.setValue(28)
        self.brush_size.setButtonSymbols(QSpinBox.NoButtons); self.brush_size.setAlignment(Qt.AlignCenter)
        self.brush_size.setSuffix(' px'); self.brush_size.setFixedSize(55,28)
        self.brush_size.setToolTip('빠른 선택 원형 커서의 크기'); self.brush_size.valueChanged.connect(self.set_brush_size)
        brush_layout.addWidget(self.brush_size)
        self.brush_plus = self.button('+', lambda: self.brush_size.stepUp()); self.brush_plus.setFixedSize(26,28)
        self.brush_plus.setToolTip('선택 크기 키우기'); brush_layout.addWidget(self.brush_plus)
        mode_row.addWidget(self.brush_control)
        self.transform_options=QFrame(); self.transform_options.setStyleSheet('QFrame { background: transparent; border: none; } QSpinBox, QDoubleSpinBox { background:#293442; color:#eff4fc; border:1px solid #536075; border-radius:4px; padding:4px; } QCheckBox { background:transparent; color:#e8edf5; }')
        transform_row=QHBoxLayout(self.transform_options);transform_row.setContentsMargins(0,0,0,0);transform_row.setSpacing(5)
        self.transform_x=QSpinBox();self.transform_y=QSpinBox()
        for label,box in [('X',self.transform_x),('Y',self.transform_y)]:
            transform_row.addWidget(ScrubLabel(label,box));box.setRange(-50000,50000);box.setSuffix(' px');box.setButtonSymbols(QSpinBox.NoButtons);box.setFixedWidth(80);transform_row.addWidget(box)
        self.transform_w=QDoubleSpinBox();self.transform_h=QDoubleSpinBox()
        for label,box in [('W',self.transform_w),('H',self.transform_h)]:
            transform_row.addWidget(ScrubLabel(label,box));box.setRange(0.1,10000);box.setDecimals(1);box.setSuffix(' %');box.setButtonSymbols(QDoubleSpinBox.NoButtons);box.setFixedWidth(85);transform_row.addWidget(box)
        self.transform_ratio=QCheckBox('비율 유지');self.transform_ratio.setChecked(True);transform_row.addWidget(self.transform_ratio)
        self.transform_x.valueChanged.connect(lambda value:self.transform_input_changed('x'))
        self.transform_y.valueChanged.connect(lambda value:self.transform_input_changed('y'))
        self.transform_w.valueChanged.connect(lambda value:self.transform_input_changed('w'))
        self.transform_h.valueChanged.connect(lambda value:self.transform_input_changed('h'))
        self.transform_ratio.toggled.connect(self.transform_ratio_changed)
        self.transform_options.hide();mode_row.addWidget(self.transform_options)
        self.transform_apply_btn=self.button('✓ 적용', self.apply_transform);self.transform_apply_btn.hide();mode_row.addWidget(self.transform_apply_btn)
        self.transform_cancel_btn=self.button('취소', self.cancel_transform);self.transform_cancel_btn.hide();mode_row.addWidget(self.transform_cancel_btn)
        mode_row.addStretch(); toolbar_layout.addLayout(mode_row)
        outer.addWidget(toolbar)
        self.tabs = QTabBar(); self.tabs.setTabsClosable(False); self.tabs.setMovable(False); self.tabs.setExpanding(False)
        self.tabs.currentChanged.connect(self.switch_document); self.tabs.currentChanged.connect(self.defer_active_tab_margin); self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        tab_row = QFrame(); tab_row.setObjectName('tabRow')
        tab_layout = QHBoxLayout(tab_row); tab_layout.setContentsMargins(0,0,4,0); tab_layout.setSpacing(4)
        self.empty_tab = QFrame(); self.empty_tab.setObjectName('emptyTab'); self.empty_tab.setFixedSize(38, 32)
        tab_layout.addWidget(self.empty_tab)
        tab_layout.addWidget(self.tabs); self.tabs.hide()
        self.new_tab_btn = self.button('+', self.new_tab); self.new_tab_btn.setObjectName('newTabButton')
        self.new_tab_btn.setFixedSize(30,30); self.new_tab_btn.setToolTip('새 탭 (Ctrl+Shift+N)'); self.new_tab_btn.setAccessibleName('새 탭')
        tab_layout.addWidget(self.new_tab_btn); tab_layout.addStretch()
        outer.addWidget(tab_row)
        main = QHBoxLayout(); main.setSpacing(0); outer.addLayout(main, 1)
        side = QFrame(); side.setObjectName('side'); side.setFixedWidth(55)
        main.addWidget(side)
        col = QVBoxLayout(side); col.setContentsMargins(6, 8, 6, 8); col.setSpacing(4)
        self.move_btn = self.icon_button('move', '이동 도구 (V)', self.move_tool); self.move_btn.setCheckable(True); col.addWidget(self.move_btn)
        self.rect_btn = self.icon_button('rectangle', '사각형 선택', self.rectangle_tool); self.rect_btn.setCheckable(True); self.rect_btn.setChecked(True); col.addWidget(self.rect_btn)
        self.lasso_btn = self.icon_button('lasso', '올가미 선택', self.lasso_tool); self.lasso_btn.setCheckable(True); col.addWidget(self.lasso_btn)
        self.auto_side_btn = self.icon_button('subject', '인물 자동 선택', self.auto); col.addWidget(self.auto_side_btn)
        self.auto_objects_btn = self.icon_button('object', '사물 자동 선택', self.auto_objects); col.addWidget(self.auto_objects_btn)
        self.object_btn = self.icon_button('object', '객체 선택', self.object_tool); self.object_btn.setCheckable(True); col.addWidget(self.object_btn)
        self.brush_btn = self.icon_button('brush', '빠른 선택', self.brush_tool); self.brush_btn.setCheckable(True); col.addWidget(self.brush_btn)
        self.wand_btn = self.icon_button('wand', '자동 선택', self.wand_tool); self.wand_btn.setCheckable(True); col.addWidget(self.wand_btn)
        col.addStretch()
        self.canvas = Canvas()
        self.scroll = QScrollArea(); self.scroll.setWidget(self.canvas); self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.workspace_splitter=QSplitter(Qt.Horizontal);self.workspace_splitter.setHandleWidth(6);self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(self.scroll);main.addWidget(self.workspace_splitter,1)
        self.canvas.zoom_requested.connect(self.zoom_at_pointer)
        self.canvas.object_requested.connect(self.select_object)
        self.canvas.point_requested.connect(self.select_point)
        self.canvas.move_requested.connect(self.move_layer)
        self.canvas.layer_chosen.connect(self.select_canvas_layer)
        self.canvas.layer_resize_requested.connect(self.resize_selected_layer)
        self.canvas.transform_geometry_changed.connect(self.update_transform_inputs)
        self.canvas.transform_geometry_changed.connect(self.update_transform_bar)
        self.canvas.transform_geometry_changed.connect(self.update_transform_history_buttons)
        self.scroll.viewport().installEventFilter(self)
        self.selection_bar = QFrame(self.scroll.viewport()); self.selection_bar.setObjectName('selectionBar')
        self.selection_bar.setStyleSheet('QFrame#selectionBar { background: #3b3d42; border: 1px solid #72757b; border-radius: 5px; } QPushButton { padding: 5px 10px; border-radius: 3px; }')
        bar_layout = QHBoxLayout(self.selection_bar); bar_layout.setContentsMargins(5, 4, 5, 4); bar_layout.setSpacing(5)
        self.subject_button=self.button('피사체 선택',lambda:None)
        self.subject_button.setToolTip('사람 또는 사물을 선택해 자동으로 선택합니다')
        self.subject_menu=QMenu(self.subject_button);self.subject_menu.setStyleSheet(self.file_menu.styleSheet())
        self.subject_person_action=self.subject_menu.addAction('사람');self.subject_person_action.triggered.connect(lambda:self.run(True,'person'))
        self.subject_object_action=self.subject_menu.addAction('사물');self.subject_object_action.triggered.connect(lambda:self.run(True,'objects'))
        self.subject_button.setMenu(self.subject_menu);bar_layout.addWidget(self.subject_button)
        for label, action, tip in [('복사', self.copy_selection, '선택한 레이어의 영역을 복사 (Ctrl+C)'), ('선택 부분 제거', self.delete_selection, '선택 영역을 투명하게 삭제'), ('누끼따기', self.cut, '선택한 피사체만 남기기'), ('선택 해제', self.deselect, '선택 해제 (Ctrl+D)')]:
            btn = self.button(label, action); btn.setToolTip(tip); bar_layout.addWidget(btn)
        self.selection_bar.adjustSize(); self.selection_bar.hide()
        self.transform_bar=QFrame(self.scroll.viewport());self.transform_bar.setObjectName('transformBar')
        self.transform_bar.setStyleSheet('QFrame#transformBar { background:#3b3d42; border:1px solid #72757b; border-radius:6px; } QPushButton { padding:6px 14px; border-radius:4px; }')
        transform_bar_layout=QHBoxLayout(self.transform_bar);transform_bar_layout.setContentsMargins(5,5,5,5);transform_bar_layout.setSpacing(6)
        self.transform_bottom_cancel=self.button('취소',self.cancel_transform);self.transform_bottom_cancel.setToolTip('변형 취소 (Esc)')
        self.transform_bottom_apply=self.button('적용',self.apply_transform,primary=True);self.transform_bottom_apply.setToolTip('변형 적용 (Enter)')
        transform_bar_layout.addWidget(self.transform_bottom_cancel);transform_bar_layout.addWidget(self.transform_bottom_apply)
        self.transform_bar.adjustSize();self.transform_bar.hide()
        self.loading_panel = QFrame(self.scroll.viewport()); self.loading_panel.setObjectName('loadingPanel')
        self.loading_panel.setStyleSheet('''
            QFrame#loadingPanel { background: #252d3b; border: 1px solid #52647d; border-radius: 10px; }
            QLabel { background: transparent; border: none; color: #f3f6fc; }
            QLabel#loadingTitle { font-size: 15px; font-weight: bold; }
            QLabel#loadingHint { font-size: 12px; color: #aebdd1; }
            QProgressBar { border: none; border-radius: 3px; background: #3a4659; height: 6px; }
            QProgressBar::chunk { border-radius: 3px; background: #8296ff; }
        ''')
        loading_layout = QVBoxLayout(self.loading_panel); loading_layout.setContentsMargins(20,18,20,18); loading_layout.setSpacing(9)
        self.loading_title = QLabel('사진을 분석하고 있습니다'); self.loading_title.setObjectName('loadingTitle'); self.loading_title.setWordWrap(True); loading_layout.addWidget(self.loading_title)
        self.loading_hint = QLabel('처리 중입니다. 잠시만 기다려 주세요.'); self.loading_hint.setObjectName('loadingHint'); self.loading_hint.setWordWrap(True); loading_layout.addWidget(self.loading_hint)
        self.loading_progress = QProgressBar(); self.loading_progress.setRange(0,0); self.loading_progress.setTextVisible(False); self.loading_progress.setFixedHeight(6); loading_layout.addWidget(self.loading_progress)
        self.loading_panel.hide()
        right = QFrame(); self.right_panel=right;right.setObjectName('side');right.setMinimumWidth(200);right.setMaximumWidth(600)
        rcol = QVBoxLayout(right); rcol.setContentsMargins(14, 18, 14, 14)
        label = self.section('내비게이터'); rcol.addWidget(label)
        self.navigator = Navigator(); self.navigator.pan_requested.connect(self.pan_from_navigator); self.navigator.setAlignment(Qt.AlignCenter);self.navigator.setMinimumWidth(160);self.navigator.setFixedHeight(150);self.navigator.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.navigator.setStyleSheet('background: #333943; border: 1px solid #505866;'); rcol.addWidget(self.navigator)
        self.zoom_label = QLabel('100%'); rcol.addWidget(self.zoom_label)
        self.zoom_slider = QSlider(Qt.Horizontal); self.zoom_slider.setRange(25, 400); self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.set_zoom); rcol.addWidget(self.zoom_slider)
        buttons = QHBoxLayout()
        minus = self.button('−', lambda: self.zoom_step(-1)); plus = self.button('+', lambda: self.zoom_step(1))
        fit = self.button('맞춤', self.fit_zoom); buttons.addWidget(minus); buttons.addWidget(fit); buttons.addWidget(plus); rcol.addLayout(buttons)
        rcol.addSpacing(15)
        rcol.addSpacing(15)
        rcol.addWidget(self.section('레이어'))
        self.layer_list = QListWidget(); self.layer_list.setMinimumHeight(150)
        self.layer_list.setStyleSheet('QListWidget { background: #252d3a; color: #e6edf7; border: 1px solid #465266; border-radius: 5px; outline: none; } QListWidget::item { padding: 7px; border-bottom: 1px solid #394556; } QListWidget::item:selected { background: #526482; color: white; }')
        self._updating_layers = False; self._layer_list_key = None; self._layer_thumbnail_cache = {}
        self.layer_list.currentRowChanged.connect(self.choose_layer)
        self.layer_list.itemChanged.connect(self.toggle_layer_visibility)
        rcol.addWidget(self.layer_list, 1);self.workspace_splitter.addWidget(right)
        self.workspace_splitter.setStretchFactor(0,1);self.workspace_splitter.setStretchFactor(1,0)
        self.workspace_splitter.setSizes([1000,220]);self.workspace_splitter.splitterMoved.connect(lambda *_:self.update_navigator())
        self.scroll.horizontalScrollBar().valueChanged.connect(self.update_navigator)
        self.scroll.verticalScrollBar().valueChanged.connect(self.update_navigator)
        self.transform_shortcut = QShortcut(QKeySequence('Ctrl+T'), self); self.transform_shortcut.activated.connect(self.transform_layer_dialog)
        self.move_shortcut = QShortcut(QKeySequence('V'), self); self.move_shortcut.activated.connect(self.move_tool)
        self.rectangle_shortcut = QShortcut(QKeySequence('M'), self); self.rectangle_shortcut.activated.connect(self.rectangle_tool)
        self.lasso_shortcut = QShortcut(QKeySequence('L'), self); self.lasso_shortcut.activated.connect(self.lasso_tool)
        self.transform_accept_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self); self.transform_accept_shortcut.activated.connect(self.apply_transform); self.transform_accept_shortcut.setEnabled(False)
        self.transform_enter_shortcut = QShortcut(QKeySequence(Qt.Key_Enter), self); self.transform_enter_shortcut.activated.connect(self.apply_transform); self.transform_enter_shortcut.setEnabled(False)
        self.transform_cancel_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self); self.transform_cancel_shortcut.activated.connect(self.cancel_transform); self.transform_cancel_shortcut.setEnabled(False)
        self.copy_shortcut = QShortcut(QKeySequence.Copy, self); self.copy_shortcut.activated.connect(self.copy_selection)
        self.paste_shortcut = QShortcut(QKeySequence.Paste, self); self.paste_shortcut.activated.connect(self.paste_image)
        self.shortcut = QShortcut(QKeySequence.Undo, self); self.shortcut.activated.connect(self.undo)
        self.redo_shortcut = QShortcut(QKeySequence('Ctrl+Y'), self); self.redo_shortcut.activated.connect(self.redo)
        self.redo_shortcut_alt = QShortcut(QKeySequence('Ctrl+Shift+Z'), self); self.redo_shortcut_alt.activated.connect(self.redo)
        self.delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self); self.delete_shortcut.activated.connect(self.handle_delete)
        self.canvas.changed.connect(self.canvas_changed)
        self.update_sidebar_mode(False)
        self.brush_control.hide(); self.brush_size_label.hide()
        status_row = QHBoxLayout(); status_row.setContentsMargins(0,0,0,0); status_row.setSpacing(0)
        self.size_status = StatusLabel('크기 없음'); self.size_status.setObjectName('status'); status_row.addWidget(self.size_status)
        self.status = StatusLabel('사진을 열어 시작하세요.'); self.status.setObjectName('status'); status_row.addWidget(self.status,1); outer.addLayout(status_row)
        self.canvas.history_budget = self.preferences['history_mb'] * 1024 * 1024
        self.toast_label=QLabel(self); self.toast_label.setAttribute(Qt.WA_TransparentForMouseEvents); self.toast_label.hide()
        self.toast_label.setStyleSheet('QLabel { background: #293442; color: #f4f7fc; border: 1px solid #7188a6; border-radius: 8px; padding: 9px 16px; font-weight: 600; }')
        self.toast_animation=QPropertyAnimation(self.toast_label,b'pos',self);self.toast_animation.setDuration(180);self.toast_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.toast_timer=QTimer(self);self.toast_timer.setSingleShot(True);self.toast_timer.timeout.connect(self.toast_label.hide)
        for button in self.findChildren(AnimatedButton):
            if button.icon().isNull(): continue
            button.clicked.connect(lambda checked=False, target=button:self.show_action_toast(target.toolTip().split(' (')[0]))
        for name, shortcut in [('새 문서',self.shortcut_actions['new_document']),('사진 불러오기',self.shortcut_actions['open']),('저장',self.shortcut_actions['save'])]:
            shortcut.triggered.connect(lambda checked=False, text=name:self.show_action_toast(text))
        for name, shortcut in [('실행 취소',self.shortcut),('다시 실행',self.redo_shortcut),('다시 실행',self.redo_shortcut_alt),('복사',self.copy_shortcut),('붙여넣기',self.paste_shortcut),('레이어 변형',self.transform_shortcut),('이동 도구',self.move_shortcut),('사각형 선택',self.rectangle_shortcut),('올가미 선택',self.lasso_shortcut),('삭제',self.delete_shortcut)]:
            shortcut.activated.connect(lambda text=name:self.show_action_toast(text))
        self._dark_window_style = self.styleSheet()
        self._dark_inline_styles = [(widget, widget.styleSheet()) for widget in self.findChildren(QWidget) if widget.styleSheet()]
        self.apply_shortcuts()
        self.set_enabled()
        self.apply_theme()
        translate_widgets(self,self.preferences['language'])
        self.show_tool_tolerance(self.canvas.tool)
        self.file_name.installEventFilter(self)
        for numeric in self.findChildren(QSpinBox)+self.findChildren(QDoubleSpinBox):
            numeric.installEventFilter(self);numeric.lineEdit().installEventFilter(self)
        QTimer.singleShot(0, self.apply_theme)
    def defer_active_tab_margin(self, index):
        # QTabBar scrolls its current tab into view after emitting this signal.
        QTimer.singleShot(0, self.adjust_active_tab_margin)
    def adjust_active_tab_margin(self):
        index=self.tabs.currentIndex()
        if index<=0:return
        buttons=[button for button in self.tabs.findChildren(QToolButton) if button.isVisible()]
        left=next((button for button in buttons if button.arrowType()==Qt.LeftArrow),None)
        if left is None or not left.isEnabled():return
        right_edge=min((button.geometry().left() for button in buttons),default=self.tabs.width())
        # Do not move a wide selected tab underneath the scroll controls.
        for _ in range(4):
            rect=self.tabs.tabRect(index)
            if rect.left()>=24 or rect.right()+32>=right_edge or not left.isEnabled():break
            before=rect.left();left.click()
            if self.tabs.tabRect(index).left()<=before:break
    def show_action_toast(self, text):
        if not text: return
        self.toast_animation.stop();self.toast_timer.stop()
        self.toast_label.setText(ENGLISH.get(text,text) if self.preferences['language']=='en' else text);self.toast_label.adjustSize()
        x=max(0,(self.width()-self.toast_label.width())//2)
        end_y=max(0,self.height()-self.toast_label.height()-58)
        self.toast_label.move(x,end_y+18);self.toast_label.show();self.toast_label.raise_()
        self.toast_animation.setStartValue(self.toast_label.pos())
        from PyQt5.QtCore import QPoint
        self.toast_animation.setEndValue(QPoint(x,end_y));self.toast_animation.start();self.toast_timer.start(1400)
    def rename_current_tab(self, name):
        index = self.active_document
        if index < 0 or index >= len(self.documents): return
        self.documents[index]['name'] = name
        self.tabs.setTabText(index, name)
        self.status.setText(f'탭 이름을 {name}(으)로 변경했습니다.')
    def clear_temporary_cache(self):
        self.canvas.quick_lab = None
        self.canvas._alpha_bbox_cache.clear()
        self.canvas._move_pixmap_cache = None
        self.canvas._layer_preview_cache.clear();self.canvas._layer_pixmap_cache.clear();self._layer_thumbnail_cache.clear()
        QPixmapCache.clear(); gc.collect()
        self.status.setText('임시 캐시를 비웠습니다. 사진과 실행 취소 기록은 유지됩니다.')
    def open_settings(self):
        if self.canvas.busy: return
        dialog=SettingsDialog(self.preferences,self);self.apply_dialog_theme(dialog)
        if dialog.exec_()!=QDialog.Accepted: return
        try: save_settings(dialog.values)
        except OSError as error:
            QMessageBox.critical(self,'설정 저장 오류',str(error));return
        self.preferences=dialog.values
        global _RUNTIME_MEMORY_SAVING
        _RUNTIME_MEMORY_SAVING=self.preferences['memory_saving']
        self.canvas.history_budget=self.preferences['history_mb']*1024*1024
        self.capture_document()
        for doc in self.documents: trim_history(doc.get('history',[]),doc.get('future',[]),self.canvas.history_budget)
        self.apply_shortcuts();self.apply_theme();translate_widgets(self,self.preferences['language']);self.show_tool_tolerance(self.canvas.tool);self.set_enabled();self.status.refresh_language();self.size_status.refresh_language()
        self.status.setText('설정을 저장했습니다.')
    def apply_shortcuts(self):
        keys=self.preferences['shortcuts']
        self.paste_menu_action.setShortcut(QKeySequence(keys['paste']))
        for name, action in self.shortcut_actions.items(): action.setShortcut(QKeySequence(keys[name]))
        for name, shortcut in [('undo',self.shortcut),('redo',self.redo_shortcut),('copy',self.copy_shortcut),('paste',self.paste_shortcut),('transform',self.transform_shortcut),('move',self.move_shortcut),('rectangle',self.rectangle_shortcut),('lasso',self.lasso_shortcut)]: shortcut.setKey(QKeySequence(keys[name]))
        self.undo_btn.setToolTip('실행 취소 ('+keys['undo']+')');self.redo_btn.setToolTip('다시 실행 ('+keys['redo']+')')
        self.rect_btn.setToolTip('사각형 선택 ('+keys['rectangle']+')');self.lasso_btn.setToolTip('올가미 선택 ('+keys['lasso']+')')
    def apply_dialog_theme(self, dialog):
        light=self.preferences['theme']=='light'
        arrow_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'assets','chevron-'+('light' if light else 'dark')+'.svg').replace('\\','/')
        translate_widgets(dialog,self.preferences['language'])
        if light:
            for widget in [dialog]+dialog.findChildren(QWidget):
                if widget.styleSheet(): widget.setStyleSheet(themed_stylesheet(widget.styleSheet(),'light'))
            if isinstance(dialog,SettingsDialog):
                for label in dialog.findChildren(QLabel):
                    label.setStyleSheet(label.styleSheet()+'; color:#253246; background:transparent;')
                for checkbox in dialog.findChildren(QCheckBox):
                    checkbox.setStyleSheet('color:#253246; background:transparent;')
                for field in dialog.findChildren(QSpinBox):
                    field.setStyleSheet('QSpinBox { background:#ffffff; color:#253246; border:1px solid #b2c0d0; border-radius:5px; padding:7px; }')
                for button in dialog.findChildren(QPushButton):
                    if button.objectName()=='primary': button.setStyleSheet('QPushButton { background:#526ff0; color:white; border:0; border-radius:5px; padding:8px 13px; } QPushButton:hover { background:#6782ff; }')
                    else: button.setStyleSheet('QPushButton { background:#e6ebf2; color:#253246; border:1px solid #b2c0d0; border-radius:5px; padding:8px 13px; } QPushButton:hover { background:#d1def0; }')
                for field in dialog.findChildren(QKeySequenceEdit):
                    field.setStyleSheet('QKeySequenceEdit { background:#ffffff; color:#253246; border:1px solid #b2c0d0; border-radius:5px; padding:7px; } QKeySequenceEdit:focus { border:1px solid #526ff0; }')
        if light and isinstance(dialog,SelectionRadiusDialog):
            for button in dialog.findChildren(QPushButton):
                if button.objectName()=='primary':
                    button.setStyleSheet('QPushButton { background:#526ff0; color:#ffffff; border:0; border-radius:5px; padding:7px 14px; } QPushButton:hover { background:#6782ff; }')
        for combo in dialog.findChildren(QComboBox):
            if not isinstance(combo.view(), QListView): combo.setView(QListView())
            bg,fg,border,hover=('#ffffff','#253246','#b2c0d0','#e9effa') if light else ('#293442','#eff4fc','#536075','#40516b')
            combo.view().setStyleSheet(f'QListView {{ background:{bg}; color:{fg}; border:1px solid {border}; outline:0; padding:4px; selection-background-color:#526ff0; selection-color:#ffffff; }} QListView::item {{ min-height:30px; padding:2px 9px; }} QListView::item:selected {{ background:#526ff0; color:#ffffff; }}')
            combo.setStyleSheet(f'''QComboBox {{ background:{bg}; color:{fg}; border:1px solid {border}; border-radius:6px; padding:8px 30px 8px 10px; }}
                QComboBox:hover, QComboBox:on {{ border:1px solid #8296ff; }} QComboBox::drop-down {{ border:0; width:26px; }} QComboBox::down-arrow {{ image: url("{arrow_path}"); width: 14px; height: 14px; }}
                QComboBox QAbstractItemView {{ background:{bg}; color:{fg}; border:1px solid {border}; border-radius:6px; padding:4px; outline:0; selection-background-color:#526ff0; selection-color:#ffffff; }}
                QComboBox QAbstractItemView::item {{ min-height:30px; padding:2px 9px; border-radius:4px; }} QComboBox QAbstractItemView::item:hover:!selected {{ background:{hover}; }} QComboBox QAbstractItemView::item:selected {{ background:#526ff0; color:#ffffff; }}''')
        if isinstance(dialog,SettingsDialog):
            if light:
                dialog.history_control.setStyleSheet('QFrame#historyControl { background:#ffffff; border:1px solid #b2c0d0; border-radius:7px; }')
                dialog.history_limit.setStyleSheet('QSpinBox { background:transparent; color:#253246; border:0; font-weight:600; }')
            else:
                dialog.history_control.setStyleSheet('QFrame#historyControl { background:#252c38; border:1px solid #536075; border-radius:7px; }')
                dialog.history_limit.setStyleSheet('QSpinBox { background:transparent; color:#eff4fc; border:0; font-weight:600; }')
            bg,fg,border,hover=('#ffffff','#253246','#b2c0d0','#e9effa') if light else ('#293442','#eff4fc','#536075','#40516b')
            combo_style=f'''QComboBox {{ background:{bg}; color:{fg}; border:1px solid {border}; border-radius:6px; padding:8px 30px 8px 10px; }}
                QComboBox:hover, QComboBox:on {{ border:1px solid #8296ff; }} QComboBox::drop-down {{ border:0; width:26px; }} QComboBox::down-arrow {{ image: url("{arrow_path}"); width: 14px; height: 14px; }}
                QComboBox QAbstractItemView {{ background:{bg}; color:{fg}; border:1px solid {border}; border-radius:6px; padding:4px; outline:0; selection-background-color:#526ff0; selection-color:#ffffff; }}
                QComboBox QAbstractItemView::item {{ min-height:30px; padding:2px 9px; border-radius:4px; }}
                QComboBox QAbstractItemView::item:hover:!selected {{ background:{hover}; }} QComboBox QAbstractItemView::item:selected {{ background:#526ff0; color:#ffffff; }}'''
            dialog.theme.setStyleSheet(combo_style);dialog.language.setStyleSheet(combo_style)
    def apply_theme(self):
        theme=self.preferences['theme']
        set_native_titlebar_theme(self,theme=='dark')
        tip='QToolTip { color: #253246; background-color: #ffffff; border: 1px solid #b2c0d0; border-radius: 5px; padding: 7px 10px; font-size: 12px; }' if theme=='light' else 'QToolTip { color: #f4f7fc; background-color: #2a3342; border: 1px solid #566780; border-radius: 5px; padding: 7px 10px; font-size: 12px; }'
        QApplication.instance().setStyleSheet(tip)
        self.setStyleSheet(themed_stylesheet(self._dark_window_style,theme))
        for widget, css in self._dark_inline_styles:
            try: widget.setStyleSheet(themed_stylesheet(css,theme))
            except RuntimeError: pass
        arrow_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'assets','chevron-'+('light' if theme=='light' else 'dark')+'.svg').replace('\\','/')
        if theme == 'light':
            combo_bg, combo_text, combo_border, hover, selected = '#ffffff', '#253246', '#b2c0d0', '#e9effa', '#526ff0'
        else:
            combo_bg, combo_text, combo_border, hover, selected = '#2c3544', '#eef3fa', '#536075', '#40516b', '#526ff0'
        menu_bg,menu_text,menu_border,menu_hover,menu_disabled = (('#ffffff','#253246','#b2c0d0','#e9effa','#9aa9bb') if theme=='light' else ('#303949','#f4f7fc','#536075','#3b485d','#778391'))
        menu_style=f'''QPushButton {{ background:{menu_bg}; color:{menu_text}; border:1px solid {menu_border}; border-radius:5px; padding:5px 12px 5px 10px; min-width:55px; text-align:left; }}
            QPushButton:hover {{ background:{menu_hover}; border-color:#8296ff; }}
            QPushButton:pressed, QPushButton:open {{ background:{menu_hover}; }}
            QPushButton:disabled {{ background:{menu_bg}; color:{menu_disabled}; border-color:{menu_border}; }}
            QPushButton::menu-indicator {{ image:url("{arrow_path}"); width:14px; height:14px; subcontrol-origin:padding; subcontrol-position:right center; right:7px; }}'''
        self.file_btn.setStyleSheet(menu_style);self.image_btn.setStyleSheet(menu_style);self.select_btn.setStyleSheet(menu_style)
        for menu in (self.export_menu,self.select_menu,self.modify_menu):menu.setStyleSheet(self.file_menu.styleSheet())
        self.quality_box.view().setStyleSheet(f'QListView {{ background:{combo_bg}; color:{combo_text}; border:1px solid {combo_border}; outline:0; padding:4px; selection-background-color:#526ff0; selection-color:#ffffff; }} QListView::item {{ min-height:30px; padding:2px 9px; }} QListView::item:selected {{ background:#526ff0; color:#ffffff; }}')
        self.quality_box.setStyleSheet(f'''QComboBox {{ background:{combo_bg}; color:{combo_text}; border:1px solid {combo_border}; border-radius:6px; padding:8px 30px 8px 10px; }}
            QComboBox:hover, QComboBox:on {{ border:1px solid #8296ff; }}
            QComboBox::drop-down {{ border:0; width:26px; }} QComboBox::down-arrow {{ image: url("{arrow_path}"); width: 14px; height: 14px; }}
            QComboBox QAbstractItemView {{ background:{combo_bg}; color:{combo_text}; border:1px solid {combo_border}; border-radius:6px; padding:4px; outline:0; selection-background-color:{selected}; selection-color:#ffffff; }}
            QComboBox QAbstractItemView::item {{ min-height:30px; padding:2px 9px; border-radius:4px; }}
            QComboBox QAbstractItemView::item:hover:!selected {{ background:{hover}; }}
            QComboBox QAbstractItemView::item:selected {{ background:{selected}; color:#ffffff; }}''')
        self.subject_button.setStyleSheet(menu_style);self.subject_menu.setStyleSheet(self.file_menu.styleSheet())
        self.canvas.light_theme=(theme=='light');self.canvas.update()
        self.mode_switch.light_theme=(theme=='light');self.mode_switch.english=(self.preferences['language']=='en');self.mode_switch.update()
        for i in range(self.tabs.count()):
            close=self.tabs.tabButton(i,QTabBar.RightSide)
            if close: close.setIcon(tool_icon('close','#253246' if theme=='light' else '#e6edf8'))
        color='#253246' if theme=='light' else '#e6edf8'
        for name,kind in [('save_btn','save'),('undo_btn','undo'),('redo_btn','redo'),('move_btn','move'),('rect_btn','rectangle'),('lasso_btn','lasso'),('object_btn','object'),('brush_btn','brush'),('wand_btn','wand'),('auto_side_btn','subject'),('auto_objects_btn','object')]:
            button=getattr(self,name,None)
            if button is not None: button.setIcon(tool_icon(kind,color))
    def section(self, text):
        label = QLabel(text); label.setObjectName('section'); return label
    def button(self, text, action, primary=False):
        b = AnimatedButton(text); b.clicked.connect(action)
        if primary: b.setObjectName('primary')
        return b
    def icon_button(self, kind, tip, action):
        b = self.button('', action)
        b.setIcon(tool_icon(kind)); b.setIconSize(QSize(26, 26)); b.setFixedSize(42, 42)
        b.setToolTip(tip); b.setAccessibleName(tip)
        return b
    def set_enabled(self, busy=False):
        loaded = self.original is not None
        self.canvas.busy = busy
        actual_busy = busy
        transforming = self.canvas.transform is not None
        busy = busy or transforming
        for widget in (self.file_btn,self.image_btn,self.select_btn,self.mode_switch,self.tolerance_label,self.tolerance_control,self.quality_label,self.quality_box):
            widget.setVisible(not transforming)
        self.brush_size_label.setVisible(not transforming and self.canvas.tool == 'brush')
        self.brush_control.setVisible(not transforming and self.canvas.tool == 'brush')
        self.transform_options.setVisible(transforming)
        self.transform_apply_btn.setVisible(transforming)
        self.transform_cancel_btn.setVisible(transforming)
        self.transform_accept_shortcut.setEnabled(transforming and not actual_busy)
        self.transform_enter_shortcut.setEnabled(transforming and not actual_busy)
        self.transform_cancel_shortcut.setEnabled(transforming and not actual_busy)
        self.move_shortcut.setEnabled(loaded and not busy)
        self.rectangle_shortcut.setEnabled(loaded and not busy)
        self.lasso_shortcut.setEnabled(loaded and not busy)
        self.file_btn.setEnabled(not busy)
        self.image_btn.setEnabled(loaded and not busy)
        self.transform_shortcut.setEnabled(loaded and not busy and self.canvas.layer_index >= 0)
        self.layer_list.setEnabled(loaded and not busy)
        if loaded:
            dpi = self.documents[self.active_document].get('dpi') if self.active_document >= 0 else None
            dpi_text = f' ({round(dpi[0])} ppi)' if isinstance(dpi, (tuple,list)) and dpi and dpi[0] else ''
            self.size_status.setText(f'{self.canvas.canvas_size[0]} 픽셀 × {self.canvas.canvas_size[1]} 픽셀{dpi_text}')
        else: self.size_status.setText('크기 없음')
        self.copy_shortcut.setEnabled(loaded and not busy)
        self.delete_shortcut.setEnabled(loaded and not busy)
        self.paste_shortcut.setEnabled(not busy)
        self.auto_side_btn.setEnabled(loaded and not busy)
        self.auto_objects_btn.setEnabled(loaded and not busy)
        self.save_btn.setEnabled(loaded and not busy)
        self.tabs.setEnabled(not busy)
        self.new_tab_btn.setEnabled(not busy)
        self.file_menu.setEnabled(not busy)
        self.select_btn.setEnabled(loaded and not busy)
        self.selection_actions['모두'].setEnabled(loaded and not busy)
        for label in ('선택 해제','반전'):self.selection_actions[label].setEnabled(loaded and self.canvas.selection is not None and not busy)
        self.selection_actions['다시 선택'].setEnabled(loaded and self.canvas.previous_selection is not None and not busy)
        self.modify_menu.setEnabled(loaded and self.canvas.selection is not None and not busy)
        has_exportable_tabs=any(doc.get('original') is not None and doc.get('canvas_size') for doc in self.documents)
        self.export_menu.setEnabled((loaded or has_exportable_tabs) and not busy)
        self.ppt_export_action.setEnabled(has_exportable_tabs and not busy)
        for action in (self.quick_export_action,self.export_as_action,self.export_preferences_action,self.web_export_action):
            action.setEnabled(loaded and not busy)
        self.move_btn.setEnabled(loaded and not busy)
        self.rect_btn.setEnabled(loaded and not busy)
        self.lasso_btn.setEnabled(loaded and not busy)
        self.object_btn.setEnabled(loaded and not busy)
        self.wand_btn.setEnabled(loaded and not busy)
        self.brush_btn.setEnabled(loaded and not busy)
        self.undo_btn.setEnabled(loaded and not actual_busy and (bool(self.canvas.transform['undo']) if transforming else bool(self.canvas.history)))
        self.redo_btn.setEnabled(loaded and not actual_busy and (bool(self.canvas.transform['redo']) if transforming else bool(self.canvas.future)))
        self.mode_switch.setEnabled(not busy)
        self.brush_control.setEnabled(loaded and not busy)
        self.tolerance_control.setEnabled(loaded and not busy)
        self.quality_box.setEnabled(not busy)
        self.update_selection_bar();self.update_transform_bar()
        if actual_busy:
            self.update_loading_position()
            self.loading_panel.show(); self.loading_panel.raise_()
        else:
            self.loading_panel.hide()
    def canvas_changed(self):
        if self.active_document >= 0:
            self.original = self.canvas.original
        self.set_enabled(self.canvas.busy)
        self.update_navigator()
        self.update_layer_list()
    def eventFilter(self, obj, event):
        if (event.type()==QEvent.ShortcutOverride and self.canvas.transform is not None
                and (event.matches(QKeySequence.Undo) or event.matches(QKeySequence.Redo))
                and (isinstance(obj,(QSpinBox,QDoubleSpinBox)) or (isinstance(obj,QLineEdit) and isinstance(obj.parent(),(QSpinBox,QDoubleSpinBox))))):
            event.ignore();return True
        if event.type()==QEvent.ShortcutOverride and event.matches(QKeySequence.Copy):
            if isinstance(obj,DocumentNameEdit) and obj.isReadOnly():
                event.ignore();return True
            if isinstance(obj,QSpinBox) or (isinstance(obj,QLineEdit) and isinstance(obj.parent(),QSpinBox)):
                editor=obj.lineEdit() if isinstance(obj,QSpinBox) else obj
                if not editor.hasSelectedText(): event.ignore();return True
        if hasattr(self, 'scroll') and obj is self.scroll.viewport() and event.type() == event.Resize:
            QTimer.singleShot(0, self.update_canvas_size)
            QTimer.singleShot(0, self.update_loading_position)
        return super().eventFilter(obj, event)
    def update_loading_position(self):
        if not hasattr(self, 'loading_panel'): return
        viewport = self.scroll.viewport()
        self.loading_panel.setFixedWidth(max(180, min(320, viewport.width()-24)))
        self.loading_panel.adjustSize()
        self.loading_panel.move(max(0,(viewport.width()-self.loading_panel.width())//2),
                                max(0,(viewport.height()-self.loading_panel.height())//2))
    def set_loading_message(self, text):
        self.loading_title.setText(ENGLISH.get(text,text) if self.preferences['language']=='en' else text)
        self.update_loading_position()
    def update_canvas_size(self):
        if not hasattr(self, 'scroll'): return
        viewport = self.scroll.viewport().size()
        factor = self.zoom_percent / 100.0
        self.canvas.resize(max(400, int(viewport.width()*factor)), max(350, int(viewport.height()*factor)))
        self.update_navigator()
        self.update_selection_bar()
    def set_zoom(self, value):
        viewport = self.scroll.viewport()
        if self.zoom_anchor is None:
            anchor = viewport.rect().center()
            canvas_point = self.canvas.mapFrom(viewport, anchor)
        else:
            anchor, canvas_point = self.zoom_anchor
            self.zoom_anchor = None
        old_rect = self.canvas.image_rect()
        if self.canvas.image is not None and old_rect.width() > 0 and old_rect.height() > 0:
            image_x = (canvas_point.x() - old_rect.left()) / old_rect.width()
            image_y = (canvas_point.y() - old_rect.top()) / old_rect.height()
        else:
            image_x = image_y = None
        self.zoom_percent = value
        self.zoom_label.setText(f'{value}%')
        self.update_canvas_size()
        if image_x is not None:
            new_rect = self.canvas.image_rect()
            new_point = QPointF(new_rect.left() + image_x * new_rect.width(),
                                new_rect.top() + image_y * new_rect.height()).toPoint()
            actual = self.canvas.mapTo(viewport, new_point)
            xbar = self.scroll.horizontalScrollBar(); ybar = self.scroll.verticalScrollBar()
            xbar.setValue(xbar.value() + actual.x() - anchor.x())
            ybar.setValue(ybar.value() + actual.y() - anchor.y())
            self.update_navigator()
    def zoom_at_pointer(self, direction, canvas_point):
        viewport = self.scroll.viewport()
        self.zoom_anchor = (self.canvas.mapTo(viewport, canvas_point), canvas_point)
        before = self.zoom_slider.value()
        self.zoom_step(direction)
        if self.zoom_slider.value() == before:
            self.zoom_anchor = None
    def zoom_step(self, direction):
        self.zoom_slider.setValue(max(25, min(400, self.zoom_slider.value() + direction*25)))
    def fit_zoom(self): self.zoom_slider.setValue(100)
    def update_navigator(self, *args):
        if not hasattr(self, 'navigator') or self.canvas.image is None: return
        thumb = self.canvas.pix.scaled(max(1,self.navigator.width()-2), max(1,self.navigator.height()-2), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(thumb)
        sx = thumb.width() / self.canvas.width(); sy = thumb.height() / self.canvas.height()
        barx = self.scroll.horizontalScrollBar(); bary = self.scroll.verticalScrollBar()
        # The preview viewport is expressed relative to the image, not the surrounding canvas.
        image_rect = self.canvas.image_rect()
        if image_rect.width() and image_rect.height():
            x = (barx.value()-image_rect.left()) / image_rect.width() * thumb.width()
            y = (bary.value()-image_rect.top()) / image_rect.height() * thumb.height()
            vw = self.scroll.viewport().width() / image_rect.width() * thumb.width()
            vh = self.scroll.viewport().height() / image_rect.height() * thumb.height()
            view = QRectF(x, y, vw, vh).intersected(QRectF(0, 0, thumb.width()-1, thumb.height()-1))
            self.navigator.view_rect = view
            painter.setPen(QPen(QColor('#ff4545'), 2)); painter.drawRect(view)
        painter.end(); self.navigator.setPixmap(thumb)
        if self.navigator.drag_offset is None:
            self.navigator.setCursor(Qt.OpenHandCursor)
        self.update_selection_bar()
    def update_transform_bar(self):
        if not hasattr(self,'transform_bar'):return
        if self.canvas.transform is None or self.canvas.busy:
            self.transform_bar.hide();return
        rect=self.canvas.transform_screen_rect()
        point=self.canvas.mapTo(self.scroll.viewport(),QPointF(rect.center().x(),rect.bottom()).toPoint())
        viewport=self.scroll.viewport();self.transform_bar.adjustSize()
        width,height=self.transform_bar.width(),self.transform_bar.height()
        x=max(0,min(viewport.width()-width,round(point.x()-width/2)))
        y=max(0,min(viewport.height()-height-6,point.y()+10))
        self.transform_bar.move(x,y);self.transform_bar.show();self.transform_bar.raise_()
    def update_selection_bar(self):
        self.update_transform_bar()
        if not hasattr(self, 'selection_bar'): return
        selection = self.canvas.selection
        if selection is None or self.canvas.busy or self.canvas.transform is not None:
            self.selection_bar.hide(); return
        bbox = selection.getbbox()
        if bbox is None:
            self.selection_bar.hide(); return
        rect = self.canvas.image_rect()
        if rect.isEmpty(): self.selection_bar.hide(); return
        # Map selection bounds to the visible viewport, including canvas alignment and scrolling.
        point = self.canvas.mapTo(self.scroll.viewport(),
            QPointF(rect.left() + ((bbox[0] + bbox[2]) / 2) * rect.width() / self.canvas.image.width,
                    rect.top() + bbox[3] * rect.height() / self.canvas.image.height).toPoint())
        vw = self.scroll.viewport().width(); vh = self.scroll.viewport().height()
        self.selection_bar.adjustSize()
        bw = self.selection_bar.width(); bh = self.selection_bar.height()
        x = max(0, min(vw-bw, int(point.x()-bw/2)))
        y = int(point.y()+8)
        if y + bh > vh: y = int(point.y()-bh-12)
        y = max(0, min(vh-bh, y))
        self.selection_bar.move(x, y); self.selection_bar.show(); self.selection_bar.raise_()
    def invert_selection(self):
        if self.canvas.busy or self.canvas.selection is None: return
        self.canvas.snapshot()
        self.canvas.selection = Image.fromarray(255 - np.asarray(self.canvas.selection, dtype=np.uint8))
        self.canvas.refresh(); self.set_enabled()
        self.status.setText('선택 영역을 반전했습니다.')
    def set_tolerance(self, value):
        self.tolerances[self.tolerance_key] = value
        if hasattr(self, 'canvas'):
            if self.tolerance_key == 'brush': self.canvas.tolerance = value
            if hasattr(self, 'status'):
                message = f'{self.tolerance_label.text()} {value}: applies to the next selection.' if self.preferences['language']=='en' else f'{self.tolerance_label.text()} {value}: 다음 선택부터 적용됩니다.'
                self.status.setText(message)
    def show_tool_tolerance(self, tool):
        key = tool if tool in ('brush', 'wand', 'object') else 'subject'
        self.tolerance_key = key
        if self.preferences['language'] == 'en':
            label = {'subject':'Person selection', 'brush':'Quick selection', 'wand':'Auto selection', 'object':'Object selection'}[key]
            self.tolerance_label.setText(label + ' tolerance')
            self.tolerance_control.setToolTip('Applies only to ' + label.lower() + '.')
        else:
            label = {'subject':'인물 자동', 'brush':'빠른 선택', 'wand':'자동 선택', 'object':'객체 선택'}[key]
            self.tolerance_label.setText(label + ' 허용치')
            self.tolerance_control.setToolTip(label + ' 도구에만 적용됩니다.')
        self.tolerance_box.blockSignals(True)
        self.tolerance_box.setValue(self.tolerances[key])
        self.tolerance_box.blockSignals(False)
        self.canvas.tolerance = self.tolerances['brush']
    def set_brush_size(self, value):
        if not hasattr(self, 'canvas'): return
        self.canvas.brush = value
        if self.canvas.tool == 'brush': self.canvas.update_cursor(QApplication.keyboardModifiers())
    def update_layer_list(self):
        if not hasattr(self, 'layer_list'): return
        key = (id(self.canvas.layers), self.canvas.layer_index)
        if self._layer_list_key == key: return
        self._layer_list_key = key
        self._updating_layers = True
        self.layer_list.clear()
        for index in reversed(range(len(self.canvas.layers))):
            layer = self.canvas.layers[index]
            item = QListWidgetItem(layer['name'])
            item.setData(Qt.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if layer['visible'] else Qt.Unchecked)
            source=layer['image'];key=id(source);cached=self._layer_thumbnail_cache.get(key)
            if cached is None or cached[0] is not source:
                scale=min(1.0,40/source.width,30/source.height)
                thumb=source if scale==1.0 else source.resize((max(1,round(source.width*scale)),max(1,round(source.height*scale))),R.LANCZOS,reducing_gap=2.0)
                cached=(source,QIcon(QPixmap.fromImage(qimage(thumb))))
                if len(self._layer_thumbnail_cache)>=20:self._layer_thumbnail_cache.pop(next(iter(self._layer_thumbnail_cache)))
                self._layer_thumbnail_cache[key]=cached
            item.setIcon(cached[1])
            self.layer_list.addItem(item)
            if index == self.canvas.layer_index: self.layer_list.setCurrentItem(item)
        self._updating_layers = False
    def choose_layer(self, row):
        if self._updating_layers or row < 0: return
        item = self.layer_list.item(row)
        if item is not None:
            self.canvas.layer_index = int(item.data(Qt.UserRole))
            self._layer_list_key = (id(self.canvas.layers), self.canvas.layer_index)
            self.canvas.update()
    def toggle_layer_visibility(self, item):
        if self._updating_layers or self.canvas.busy: return
        index = int(item.data(Qt.UserRole))
        if index < 0 or index >= len(self.canvas.layers): return
        visible = item.checkState() == Qt.Checked
        if self.canvas.layers[index]['visible'] == visible: return
        self.canvas.snapshot(); layers = [x.copy() for x in self.canvas.layers]
        layers[index]['visible'] = visible; self.canvas.layers = layers
        self.canvas.revision = next(_content_revisions); self.recompose_layers()
    def recompose_layers(self):
        size = self.canvas.canvas_size
        if size is None: return
        base = Image.new('RGBA', size, (0,0,0,0))
        for layer in self.canvas.layers:
            if not layer['visible']: continue
            image, x, y = layer['image'], layer['x'], layer['y']
            left,top = max(0,-x),max(0,-y)
            right,bottom = min(image.width,size[0]-x),min(image.height,size[1]-y)
            if right > left and bottom > top:
                base.alpha_composite(image.crop((left,top,right,bottom)), (x+left,y+top))
        preview = base.copy(); preview.thumbnail((2048,2048), R.LANCZOS)
        self.original = self.canvas.original = base
        self.canvas.image = preview; self.canvas.quick_lab = None
        self.canvas.marks = Image.new('L',preview.size,0)
        self.canvas.result = None
        self.canvas.refresh(); self.canvas.changed.emit()
    def transform_layer_dialog(self):
        if self.canvas.busy or self.canvas.layer_index < 0 or self.canvas.transform is not None: return
        layer=self.canvas.layers[self.canvas.layer_index]
        if not layer['visible']:
            QMessageBox.information(self,'숨겨진 레이어','선택한 레이어를 표시한 뒤 변형해 주세요.');return
        image=layer['image'];x,y=layer['x'],layer['y'];w,h=image.size;cw,ch=self.canvas.canvas_size
        # The fixed scene includes off-canvas pixels so large pasted images
        # have reachable handles instead of being clipped by the canvas.
        sx,sy=min(0,x),min(0,y);ex,ey=max(cw,x+w),max(ch,y+h)
        self.transform_old_zoom=self.zoom_slider.value()
        self.transform_old_scroll=(self.scroll.horizontalScrollBar().value(),self.scroll.verticalScrollBar().value())
        self.zoom_slider.setValue(100)
        self.canvas.transform={'rect':[float(x),float(y),float(w),float(h)],'scene':(sx,sy,max(1,ex-sx),max(1,ey-sy)),
                               'pixmap':preview_pixmap(image),'underlay':preview_pixmap(composite_layers(self.canvas.layers[:self.canvas.layer_index],self.canvas.canvas_size)),
                               'above':preview_pixmap(composite_layers(self.canvas.layers[self.canvas.layer_index+1:],self.canvas.canvas_size)),
                               'original_size':(w,h), 'keep_ratio':True, 'drag':None, 'undo':[], 'redo':[]}
        self.transform_ratio.setChecked(True)
        self.update_transform_inputs()
        self.canvas.setFocusPolicy(Qt.StrongFocus);self.canvas.setFocus();self.canvas.update();self.set_enabled(False)
        self.status.setText('모서리: 비율 유지 크기 조절 · Shift: 자유 변형 · 내부 드래그: 이동 · Enter: 적용 · Esc: 취소')
    def update_transform_inputs(self):
        state=self.canvas.transform
        if state is None: return
        x,y,w,h=state['rect'];ow,oh=state['original_size']
        for box,value in ((self.transform_x,x),(self.transform_y,y),(self.transform_w,w/ow*100),(self.transform_h,h/oh*100)):
            box.blockSignals(True);box.setValue(round(value) if isinstance(box,QSpinBox) else value);box.blockSignals(False)
    def transform_ratio_changed(self,checked):
        if self.canvas.transform is not None: self.canvas.transform['keep_ratio']=checked
    def transform_input_changed(self,which):
        state=self.canvas.transform
        if state is None: return
        ow,oh=state['original_size']
        if state['keep_ratio'] and which in ('w','h'):
            source=self.transform_w if which=='w' else self.transform_h
            other=self.transform_h if which=='w' else self.transform_w
            other.blockSignals(True);other.setValue(source.value());other.blockSignals(False)
        new_rect=[float(self.transform_x.value()),float(self.transform_y.value()),
                  max(1.0,ow*self.transform_w.value()/100),max(1.0,oh*self.transform_h.value()/100)]
        if any(abs(a-b)>0.001 for a,b in zip(state['rect'],new_rect)):
            state['undo'].append(list(state['rect']));state['redo'].clear();state['rect']=new_rect
        self.canvas.update();self.update_transform_bar();self.update_transform_history_buttons()
    def restore_transform_view(self):
        self.canvas.transform=None;self.canvas.update_cursor(QApplication.keyboardModifiers())
        self.zoom_slider.setValue(self.transform_old_zoom)
        self.scroll.horizontalScrollBar().setValue(self.transform_old_scroll[0]);self.scroll.verticalScrollBar().setValue(self.transform_old_scroll[1])
        self.canvas.refresh();self.set_enabled(False)
    def cancel_transform(self):
        if self.canvas.transform is None:return
        self.restore_transform_view();self.status.setText('변형을 취소했습니다.')
    def apply_transform(self):
        state=self.canvas.transform
        if state is None:return
        x,y,width,height=(round(v) for v in state['rect'])
        width=max(1,width);height=max(1,height)
        if width*height>50_000_000:
            QMessageBox.warning(self,'크기가 너무 큽니다','레이어는 최대 5,000만 픽셀까지 변형할 수 있습니다.');return
        try:
            layer=self.canvas.layers[self.canvas.layer_index]
            image=layer['image'] if layer['image'].size==(width,height) else layer['image'].resize((width,height),R.LANCZOS)
            self.canvas.snapshot();layers=[item.copy() for item in self.canvas.layers]
            layers[self.canvas.layer_index].update({'image':image,'x':x,'y':y});self.canvas.layers=layers
            self.canvas.selection=None;self.canvas.revision=next(_content_revisions)
            self.restore_transform_view();self.recompose_layers()
            self.status.setText('레이어 변형을 적용했습니다. Ctrl+Z로 되돌릴 수 있습니다.')
        except (MemoryError,ValueError,OSError) as error:QMessageBox.critical(self,'변형 실패',str(error))
    def resize_canvas_dialog(self):
        if self.canvas.busy or self.canvas.canvas_size is None: return
        oldw,oldh=self.canvas.canvas_size;dialog=ResizeDialog('캔버스 크기',oldw,oldh,self,aspect=False);self.apply_dialog_theme(dialog)
        if dialog.exec_()!=QDialog.Accepted:return
        width,height,_,_=dialog.values()
        self.canvas.snapshot();layers=[layer.copy() for layer in self.canvas.layers]
        for layer in layers:layer['x']+=(width-oldw)//2;layer['y']+=(height-oldh)//2
        self.canvas.layers=layers;self.canvas.canvas_size=(width,height);self.canvas.selection=None
        self.canvas.revision=next(_content_revisions);self.recompose_layers();self.update_canvas_size()
        self.status.setText('캔버스 크기를 변경했습니다.')
    def resize_image_dialog(self):
        if self.canvas.busy or self.canvas.canvas_size is None or self.canvas.transform is not None: return
        oldw, oldh = self.canvas.canvas_size
        dialog = ResizeDialog('이미지 크기', oldw, oldh, self, aspect=True); self.apply_dialog_theme(dialog)
        if dialog.exec_() != QDialog.Accepted: return
        width, height, _, _ = dialog.values()
        if (width, height) == (oldw, oldh): return
        sx, sy = width / oldw, height / oldh
        try:
            layers = []
            for layer in self.canvas.layers:
                item = layer.copy()
                neww = max(1, round(item['image'].width * sx)); newh = max(1, round(item['image'].height * sy))
                if neww * newh > 50_000_000: raise ValueError('레이어 크기가 최대 5,000만 픽셀을 초과합니다.')
                item['image'] = item['image'].resize((neww, newh), R.LANCZOS)
                item['x'] = round(item['x'] * sx); item['y'] = round(item['y'] * sy)
                layers.append(item)
            self.canvas.snapshot(); self.canvas.layers = layers; self.canvas.canvas_size = (width, height); self.canvas.selection = None
            self.canvas.revision = next(_content_revisions); self.recompose_layers(); self.update_canvas_size()
            self.status.setText(f'이미지와 캔버스 크기: {oldw} × {oldh} → {width} × {height} 픽셀')
        except (MemoryError, ValueError, OSError) as error:
            QMessageBox.critical(self, '크기 변경 실패', str(error))
    def set_tool(self, tool):
        self.brush_control.setVisible(tool == 'brush')
        self.brush_size_label.setVisible(tool == 'brush')
        self.canvas.selection_modifier = 0
        self.canvas.tool = tool
        self.show_tool_tolerance(tool)
        for name, button in (('move', self.move_btn), ('rectangle', self.rect_btn), ('lasso', self.lasso_btn), ('brush', self.brush_btn), ('object', self.object_btn), ('wand', self.wand_btn)):
            button.setChecked(name == tool)
        self.canvas.update_cursor(QApplication.keyboardModifiers())
        self.canvas.update()
    def pan_from_navigator(self, x, y):
        pix = self.navigator.pixmap()
        if pix is None or self.canvas.image is None: return
        image_rect = self.canvas.image_rect()
        # Changing a scrollbar redraws the navigator synchronously, replacing its
        # QLabel-owned pixmap. Read its dimensions before changing either bar.
        pix_width, pix_height = pix.width(), pix.height()
        if pix_width <= 0 or pix_height <= 0: return
        target_x = round(image_rect.left() + x / pix_width * image_rect.width())
        target_y = round(image_rect.top() + y / pix_height * image_rect.height())
        self.scroll.horizontalScrollBar().setValue(target_x)
        self.scroll.verticalScrollBar().setValue(target_y)
    def move_tool(self):
        if self.original is None or self.canvas.busy or self.canvas.transform is not None: return
        self.set_tool('move')
        self.canvas.update()
        self.status.setText('이동 도구: 드래그로 이동 · Shift를 누르면 가로 또는 세로로 이동 · 조절점으로 크기 변경')
    def select_canvas_layer(self, index):
        if index < 0 or index >= len(self.canvas.layers): return
        self.canvas.layer_index=index
        self.update_layer_list();self.canvas.update()
        self.status.setText(f"레이어 선택: {self.canvas.layers[index]['name']}")
    def resize_selected_layer(self, x, y, width, height):
        if self.canvas.layer_index < 0 or self.canvas.busy:return
        if width*height>50_000_000:
            QMessageBox.warning(self,'크기가 너무 큽니다','레이어는 최대 5,000만 픽셀까지 변형할 수 있습니다.');return
        try:
            layer=self.canvas.layers[self.canvas.layer_index]
            image=layer['image'].resize((width,height),R.LANCZOS)
            self.canvas.snapshot();layers=[item.copy() for item in self.canvas.layers]
            layers[self.canvas.layer_index].update({'image':image,'x':x,'y':y})
            self.canvas.layers=layers;self.canvas.selection=None
            self.canvas.revision=next(_content_revisions);self.recompose_layers()
            self.status.setText('레이어 크기를 변경했습니다. Ctrl+Z로 되돌릴 수 있습니다.')
        except (MemoryError,ValueError,OSError) as error:QMessageBox.critical(self,'변형 실패',str(error))
    def move_layer(self, dx, dy):
        if self.canvas.layer_index < 0 or self.canvas.busy: return
        self.canvas.snapshot();layers=[layer.copy() for layer in self.canvas.layers]
        layer=layers[self.canvas.layer_index];layer['x']+=dx;layer['y']+=dy
        self.canvas.layers=layers;self.canvas.selection=None
        self.canvas.revision=next(_content_revisions);self.recompose_layers()
        self.status.setText('레이어를 이동했습니다. Ctrl+Z로 되돌릴 수 있습니다.')
    def rectangle_tool(self):
        self.set_tool('rectangle')
        self.status.setText('사각형 선택: 드래그 · Shift 추가 · Alt 빼기 · Delete 삭제')
    def lasso_tool(self):
        self.set_tool('lasso')
        self.status.setText('올가미: 마우스로 원하는 영역의 경계를 그리세요. Shift 추가 · Alt 빼기')
    def object_tool(self):
        self.set_tool('object')
        self.status.setText('객체 선택: 현재 레이어에서 사람이나 사물을 사각형으로 감싸세요. Shift 추가 · Alt 빼기')
    def wand_tool(self):
        self.set_tool('wand')
        self.status.setText('자동 선택: 선택할 피사체를 클릭하세요. Shift 추가 · Alt 빼기')
    def brush_tool(self):
        self.set_tool('brush')
        self.status.setText('빠른 선택: 비슷한 색 영역을 드래그하세요. Alt를 누르면 선택에서 제외합니다.')
    def select_object(self, roi, modifiers):
        if self.canvas.busy or (self.worker is not None and self.worker.isRunning()): return
        index=self.canvas.layer_index
        if index < 0 or index>=len(self.canvas.layers): return
        layer=self.canvas.layers[index]
        if not layer['visible']:
            self.status.setText('선택한 레이어를 표시한 뒤 작업해 주세요.');return
        cw,ch=self.canvas.canvas_size;pw,ph=self.canvas.image.size;x,y=layer['x'],layer['y'];source=layer['image']
        x0,y0=max(0,x),max(0,y);x1,y1=min(cw,x+source.width),min(ch,y+source.height)
        if x0>=x1 or y0>=y1:return
        px0,py0=min(pw-1,round(x0*pw/cw)),min(ph-1,round(y0*ph/ch))
        px1,py1=max(px0+1,min(pw,round(x1*pw/cw))),max(py0+1,min(ph,round(y1*ph/ch)))
        image=source.crop((x0-x,y0-y,x1-x,y1-y)).convert('RGBA').resize((px1-px0,py1-py0),R.LANCZOS)
        local=(max(0,roi[0]-px0),max(0,roi[1]-py0),min(image.width,roi[2]-px0),min(image.height,roi[3]-py0))
        if local[2]<=local[0] or local[3]<=local[1]:
            self.status.setText('선택한 레이어 위에 사각형을 그려 주세요.');return
        self.worker=Worker(image,None,True,roi=local,modifiers=modifiers,tolerance=self.tolerances['object'],quality=self.quality_box.currentData(),result_size=(pw,ph),result_box=(px0,py0,px1,py1),subjects='objects')
        self.worker.done.connect(lambda mask:self.object_finished(mask,modifiers));self.worker.failed.connect(self.failed)
        self.worker.finished.connect(lambda:self.set_enabled(False))
        self.set_loading_message('선택한 영역을 분석하고 있습니다');self.set_enabled(True)
        self.status.setText('선택한 레이어의 영역에서 피사체를 찾고 있습니다…');self.worker.start()
    def select_point(self, point, modifiers):
        if self.canvas.busy or (self.worker is not None and self.worker.isRunning()): return
        self.worker = Worker(self.canvas.image.copy(), None, True, point=point, tolerance=self.tolerances['wand'], quality=self.quality_box.currentData())
        self.worker.done.connect(lambda mask: self.point_finished(mask, modifiers))
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(lambda: self.set_enabled(False))
        self.set_loading_message('선택한 피사체를 찾고 있습니다')
        self.set_enabled(True); self.status.setText('클릭한 피사체를 찾고 있습니다…'); self.worker.start()
    def point_finished(self, mask, modifiers):
        if not np.any(np.asarray(mask)):
            self.status.setText('클릭한 대상을 찾지 못했습니다. 빠른 선택 또는 올가미 도구를 사용해 주세요.'); return
        self.canvas.apply_selection(mask, modifiers)
        self.status.setText('클릭한 피사체를 선택했습니다.')
    def object_finished(self, mask, modifiers):
        if not np.any(np.asarray(mask)):
            self.status.setText('영역에서 피사체를 찾지 못했습니다. 대상을 여유 있게 감싸거나 빠른 선택을 사용해 주세요.'); return
        self.canvas.apply_selection(mask, modifiers)
        self.status.setText('객체를 선택했습니다. Shift로 추가, Alt로 제외할 수 있습니다.')
    def update_sidebar_mode(self, manual):
        # Common area tools remain available in both modes.
        self.auto_side_btn.setVisible(not manual)
        self.auto_objects_btn.setVisible(not manual)
        for button in (self.object_btn, self.brush_btn, self.wand_btn):
            button.setVisible(manual)
    def change_mode(self, mode):
        self.canvas.manual = bool(mode)
        self.update_sidebar_mode(bool(mode))
        if mode: self.brush_tool()
        else: self.rectangle_tool()
    def keyPressEvent(self, event):
        if self.canvas.tool == 'brush' and event.key() in (Qt.Key_BracketLeft, Qt.Key_BracketRight):
            self.brush_size.setValue(self.brush_size.value() + (-4 if event.key() == Qt.Key_BracketLeft else 4))
            event.accept(); return
        if event.key() in (Qt.Key_Shift, Qt.Key_Alt): self.canvas.update_cursor(event.modifiers())
        super().keyPressEvent(event)
    def keyReleaseEvent(self, event):
        if event.key() in (Qt.Key_Shift, Qt.Key_Alt): self.canvas.update_cursor(event.modifiers())
        super().keyReleaseEvent(event)
    def apply_selection_to_layer(self, keep):
        if self.canvas.busy or self.canvas.selection is None or self.canvas.layer_index < 0: return
        layer = self.canvas.layers[self.canvas.layer_index]
        if not layer['visible']:
            QMessageBox.information(self, '숨겨진 레이어', '선택한 레이어를 표시한 뒤 작업해 주세요.'); return
        selection = prepare_selection_mask(self.canvas.selection,self.canvas.canvas_size)
        image = layer['image'].copy()
        x,y = layer['x'],layer['y']
        local = selection.crop((x,y,x+image.width,y+image.height))
        if not keep: local = ImageChops.invert(local)
        image.putalpha(ImageChops.multiply(image.getchannel('A'), local))
        # Erase hidden source colors immediately on the edited layer. Pixels
        # with partial alpha keep their original RGB for smooth edge blending.
        pixels=np.array(image,copy=True)
        pixels[pixels[:,:,3]==0,:3]=0
        image=Image.fromarray(pixels,'RGBA')
        self.canvas.snapshot(); layers = [item.copy() for item in self.canvas.layers]
        layers[self.canvas.layer_index]['image'] = image
        self.canvas.layers = layers; self.canvas.revision = next(_content_revisions)
        if not keep: self.canvas.selection = None
        self.recompose_layers()
        self.status.setText('선택한 레이어의 배경을 제거했습니다.' if keep else '선택한 레이어에서 선택 영역을 제거했습니다.')
    def handle_delete(self):
        focused = QApplication.focusWidget()
        if isinstance(focused, QLineEdit):
            focused.del_(); return
        if isinstance(focused, (QTextEdit, QPlainTextEdit)):
            cursor = focused.textCursor()
            if cursor.hasSelection(): cursor.removeSelectedText()
            else: cursor.deleteChar()
            focused.setTextCursor(cursor); return
        if self.canvas.busy or self.canvas.transform is not None: return
        if self.canvas.selection is not None:
            self.delete_selection(); return
        if self.canvas.tool == 'move' or (focused is not None and (focused is self.layer_list or self.layer_list.isAncestorOf(focused))):
            self.delete_selected_layer()
    def delete_selected_layer(self):
        if self.canvas.busy or self.canvas.transform is not None: return
        index = self.canvas.layer_index
        if index < 0 or index >= len(self.canvas.layers): return
        name = self.canvas.layers[index]['name']
        self.canvas.snapshot()
        layers = [item.copy() for item in self.canvas.layers]
        layers.pop(index)
        self.canvas.layers = layers
        self.canvas.layer_index = min(index, len(layers)-1)
        self.canvas.selection = None
        self.canvas.revision = next(_content_revisions)
        self.recompose_layers()
        self.status.setText(f'{name} 레이어를 삭제했습니다. Ctrl+Z로 되돌릴 수 있습니다.')
    def delete_selection(self):
        self.apply_selection_to_layer(False)
    def deselect(self):
        if self.canvas.busy or self.canvas.selection is None: return
        self.canvas.snapshot(); self.canvas.previous_selection=self.canvas.selection.copy();self.canvas.selection = None
        self.canvas.refresh(); self.set_enabled()
        self.status.setText('선택을 해제했습니다.')
    def select_all(self):
        if self.canvas.busy or self.canvas.image is None or self.canvas.transform is not None:return
        self.canvas.snapshot();self.canvas.selection=Image.new('L',self.canvas.image.size,255)
        self.canvas.refresh();self.set_enabled();self.status.setText('전체를 선택했습니다.')
    def reselect(self):
        if self.canvas.busy or self.canvas.previous_selection is None or self.canvas.image is None:return
        if self.canvas.previous_selection.size!=self.canvas.image.size:
            self.status.setText('이전 선택 크기가 현재 이미지와 다릅니다.');return
        self.canvas.snapshot();self.canvas.selection=self.canvas.previous_selection.copy()
        self.canvas.refresh();self.set_enabled();self.status.setText('이전 선택을 복원했습니다.')
    def modify_selection_dialog(self, kind):
        if self.canvas.busy or self.canvas.selection is None or self.canvas.transform is not None:return
        names={'border':'테두리','smooth':'매끄럽게','expand':'확대','contract':'축소','feather':'페더'}
        dialog=SelectionRadiusDialog('선택 영역 '+names[kind],self);self.apply_dialog_theme(dialog)
        if dialog.exec_()!=QDialog.Accepted:return
        self.modify_selection(kind,dialog.radius.value())
    def modify_selection(self,kind,amount):
        if self.canvas.selection is None:return
        mask=np.asarray(self.canvas.selection.convert('L'),dtype=np.uint8)
        scale=self.canvas.image.width/self.canvas.canvas_size[0]
        radius=max(1,min(250,round(amount*scale)));size=radius*2+1
        if kind=='feather': result=cv2.GaussianBlur(mask,(0,0),max(0.1,radius/2),borderType=cv2.BORDER_CONSTANT)
        elif kind=='smooth': result=cv2.medianBlur(mask,max(3,size))
        else:
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
            if kind=='expand': result=cv2.dilate(mask,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
            elif kind=='contract':result=cv2.erode(mask,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
            elif kind=='border':
                outer=cv2.dilate(mask,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
                inner=cv2.erode(mask,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0);result=cv2.subtract(outer,inner)
            else:return
        self.canvas.snapshot();self.canvas.selection=Image.fromarray(result);self.canvas.refresh();self.set_enabled()
        self.status.setText('선택 영역을 수정했습니다.')
    def update_transform_history_buttons(self):
        state=self.canvas.transform
        if state is None:return
        self.undo_btn.setEnabled(bool(state['undo']) and not self.canvas.busy)
        self.redo_btn.setEnabled(bool(state['redo']) and not self.canvas.busy)
    def undo(self):
        state=self.canvas.transform
        if state is not None:
            if state['undo'] and state.get('drag') is None:
                state['redo'].append(list(state['rect']));state['rect']=state['undo'].pop()
                self.canvas.update();self.update_transform_inputs();self.update_transform_bar();self.update_transform_history_buttons()
                self.status.setText('현재 변형을 되돌렸습니다.')
            return
        if self.canvas.undo():
            self.status.setText('이전 작업으로 되돌렸습니다.')
            self.set_enabled()
    def redo(self):
        state=self.canvas.transform
        if state is not None:
            if state['redo'] and state.get('drag') is None:
                state['undo'].append(list(state['rect']));state['rect']=state['redo'].pop()
                self.canvas.update();self.update_transform_inputs();self.update_transform_bar();self.update_transform_history_buttons()
                self.status.setText('현재 변형을 다시 실행했습니다.')
            return
        if self.canvas.redo():
            self.status.setText('되돌린 작업을 다시 실행했습니다.')
            self.set_enabled()
    def cut(self):
        self.apply_selection_to_layer(True)
    def document_signature(self):
        result = self.canvas.result
        mask_signature = None if result is None else (result.size, result.tobytes())
        return (self.canvas.revision, mask_signature)
    def capture_document(self):
        if self.active_document < 0: return
        doc = self.documents[self.active_document]
        doc['previous_selection']=self.canvas.previous_selection
        for key in ('image', 'marks', 'result', 'selection', 'history', 'future'):
            doc[key] = getattr(self.canvas, key)
        doc['original'] = self.canvas.original
        doc['revision'] = self.canvas.revision
        doc['layers'] = self.canvas.layers
        doc['layer_index'] = self.canvas.layer_index
        doc['canvas_size'] = self.canvas.canvas_size
        doc['zoom'] = self.zoom_slider.value()
        doc['scroll'] = (self.scroll.horizontalScrollBar().value(), self.scroll.verticalScrollBar().value())
    def switch_document(self, index):
        if self.switching_document: return
        self.capture_document()
        self.active_document = index
        if index < 0:
            self.original = None
            self.canvas.image = self.canvas.original = self.canvas.marks = self.canvas.result = self.canvas.selection = None
            self.canvas.revision = 0
            self.canvas.previous_selection=None
            self.canvas.layers = []; self.canvas.layer_index = -1; self.canvas.canvas_size = None
            self.canvas.history = []; self.canvas.future = []; self.canvas.pix = None
            self.canvas.update(); self.file_name.setText('제목 없음')
            self.navigator.clear(); self.navigator.setText('사진 없음'); self.navigator.setCursor(Qt.ArrowCursor)
            self.update_navigator(); self.set_enabled(); self.update_layer_list(); return
        doc = self.documents[index]
        self.original = doc['original']
        self.canvas.original = self.original
        self.canvas.revision = doc.get('revision', 0)
        self.canvas.layers = doc.get('layers', [])
        self.canvas.layer_index = doc.get('layer_index', -1)
        self.canvas.canvas_size = doc.get('canvas_size')
        self.canvas.previous_selection=doc.get('previous_selection')
        for key in ('image', 'marks', 'result', 'selection', 'history', 'future'):
            setattr(self.canvas, key, doc[key])
        self.canvas.quick_lab = None
        self.canvas.drag_start = self.canvas.drag_current = None
        if self.canvas.image is None:
            self.canvas.pix = None; self.canvas.update()
            self.navigator.clear(); self.navigator.setText('사진 없음'); self.navigator.setCursor(Qt.ArrowCursor)
        else:
            self.canvas.refresh()
        self.file_name.setText(doc['name'])
        self.zoom_slider.setValue(doc['zoom']); self.update_canvas_size()
        self.scroll.horizontalScrollBar().setValue(doc['scroll'][0]); self.scroll.verticalScrollBar().setValue(doc['scroll'][1])
        self.update_navigator(); self.set_enabled(); self.update_layer_list()
    def new_document(self):
        if self.canvas.busy: return
        dialog = NewDocumentDialog(self)
        self.apply_dialog_theme(dialog)
        if dialog.exec_() != QDialog.Accepted: return
        name, width, height, dpi, background = dialog.settings()
        try:
            color = (255,255,255,255) if background == '흰색' else (0,0,0,0)
            image = Image.new('RGBA', (width,height), color)
            image.info['dpi'] = (dpi,dpi)
            self.load_image(image, name, force_new_tab=True)
        except (MemoryError, OSError, ValueError) as error:
            QMessageBox.critical(self, '새 문서를 만들 수 없습니다', str(error))
    def new_tab(self):
        if self.canvas.busy: return
        self.capture_document()
        doc = {'original': None, 'name': '새 탭', 'image': None, 'marks': None, 'result': None,
               'selection': None, 'history': [], 'future': [], 'zoom': 100, 'scroll': (0,0),
               'save_path': None, 'unsaved': False, 'saved_signature': (0, None), 'revision': 0, 'layers': [], 'layer_index': -1, 'canvas_size': None}
        self.documents.append(doc)
        self.switching_document = True
        index = self.tabs.addTab('새 탭'); self.tabs.show(); self.empty_tab.hide()
        self.install_tab_close_button(index)
        self.tabs.setCurrentIndex(index)
        self.switching_document = False
        self.active_document = -1
        self.switch_document(index)
        self.status.setText('파일 메뉴에서 사진을 불러오거나 Ctrl+V로 붙여넣으세요.')
    def close_current_tab(self):
        if self.active_document >= 0: self.close_tab(self.active_document)
    def close_tab(self, index):
        if self.canvas.busy or index < 0: return False
        if index != self.active_document: self.tabs.setCurrentIndex(index)
        doc = self.documents[index]
        if doc['unsaved'] or self.document_signature() != doc['saved_signature']:
            english=self.preferences['language']=='en'
            prompt = QMessageBox(QMessageBox.Question, 'Unsaved changes' if english else '저장되지 않은 변경 사항', f"Save changes to {doc['name']}?" if english else f"{doc['name']}의 변경 사항을 저장하시겠습니까?", parent=self)
            prompt.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
            save_button = prompt.addButton('Save' if english else '저장', QMessageBox.AcceptRole)
            discard_button = prompt.addButton('Discard' if english else '저장 안 함', QMessageBox.DestructiveRole)
            cancel_button = prompt.addButton('Cancel' if english else '취소', QMessageBox.RejectRole)
            prompt.setDefaultButton(save_button)
            prompt.exec_()
            clicked = prompt.clickedButton()
            answer = QMessageBox.Save if clicked is save_button else QMessageBox.Discard if clicked is discard_button else QMessageBox.Cancel
            if answer == QMessageBox.Cancel: return False
            if answer == QMessageBox.Save and not self.save(): return False
        self.switching_document = True
        self.documents.pop(index); self.tabs.removeTab(index)
        self.tabs.setVisible(self.tabs.count() > 0); self.empty_tab.setVisible(self.tabs.count() == 0)
        self.active_document = -1
        self.switching_document = False
        self.switch_document(self.tabs.currentIndex())
        return True
    def close_tab_button(self, button):
        for index in range(self.tabs.count()):
            if self.tabs.tabButton(index, QTabBar.RightSide) is button:
                self.close_tab(index)
                return
    def install_tab_close_button(self, index):
        close_button = self.button('', lambda: None); close_button.setObjectName('tabClose')
        close_button.setIcon(tool_icon('close','#253246' if self.preferences['theme']=='light' else '#e6edf8')); close_button.setIconSize(QSize(14,14)); close_button.setFixedSize(20,20)
        close_button.setToolTip('사진 탭 닫기')
        close_button.clicked.connect(lambda checked=False, button=close_button: self.close_tab_button(button))
        self.tabs.setTabButton(index, QTabBar.RightSide, close_button)
    def load_image(self, image, name, source_path=None, force_new_tab=False):
        original = ImageOps.exif_transpose(image).convert('RGBA')
        preview = original.copy()
        preview.thumbnail((2048,2048), R.LANCZOS)
        self.capture_document()
        doc = {'original': original, 'name': name, 'image': preview, 'marks': Image.new('L', preview.size, 0), 'result': None, 'selection': None, 'history': [], 'future': [], 'zoom': 100, 'scroll': (0, 0), 'save_path': None, 'unsaved': source_path is None, 'saved_signature': (0, None), 'revision': 0, 'dpi': image.info.get('dpi'), 'layers': [{'name': '배경' if source_path is None and name != '붙여넣은 사진' else name, 'image': original, 'x': 0, 'y': 0, 'visible': True}], 'layer_index': 0, 'canvas_size': original.size}
        empty = not force_new_tab and self.active_document >= 0 and self.documents[self.active_document]['original'] is None
        self.switching_document = True
        if empty:
            index = self.active_document
            self.documents[index] = doc
            self.tabs.setTabText(index, name)
        else:
            self.documents.append(doc)
            index = self.tabs.addTab(name); self.tabs.show(); self.empty_tab.hide()
            self.install_tab_close_button(index)
        self.tabs.setCurrentIndex(index)
        self.switching_document = False
        # Avoid overwriting the new document with the previous canvas state.
        self.active_document = -1
        self.switch_document(index)
        self.status.setText(f'{name}  ·  {original.width} × {original.height}  ·  원하는 작업을 선택하세요')
        self.set_enabled()
    def open(self):
        if self.canvas.busy: return
        paths, _ = QFileDialog.getOpenFileNames(self, '사진 열기', '', '이미지 (*.jpg *.jpeg *.png *.webp *.bmp *.heic *.heif);;모든 파일 (*)')
        for path in paths:
            try:
                with Image.open(path) as im:
                    self.load_image(im, os.path.basename(path), path)
            except Exception as e:
                QMessageBox.critical(self, '사진을 열 수 없습니다', f'{path}\n{e}')
    def dragEnterEvent(self, event):
        if not self.canvas.busy and self.canvas.transform is None and event.mimeData().hasUrls() and any(url.isLocalFile() and os.path.splitext(url.toLocalFile())[1].lower() in ('.jpg','.jpeg','.png','.webp','.bmp','.heic','.heif') for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else: event.ignore()
    def dragMoveEvent(self, event):
        if not self.canvas.busy and self.canvas.transform is None and event.mimeData().hasUrls(): event.acceptProposedAction()
        else: event.ignore()
    def dropEvent(self, event):
        if self.canvas.busy or self.canvas.transform is not None: event.ignore();return
        paths=[url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile() and os.path.splitext(url.toLocalFile())[1].lower() in ('.jpg','.jpeg','.png','.webp','.bmp','.heic','.heif')]
        if not paths:event.ignore();return
        for path in paths:
            try:
                with Image.open(path) as image:
                    if self.active_document<0 or self.original is None:self.load_image(image,os.path.basename(path),path)
                    else:self.paste_into_document(image,os.path.basename(path))
            except (OSError,ValueError,MemoryError) as error:
                QMessageBox.warning(self,'사진을 열 수 없습니다',f'{path}\n{error}')
        event.acceptProposedAction()
    def copy_selection(self, file_only=False):
        # Preserve ordinary text copying when an editable text field has focus.
        focused = QApplication.focusWidget()
        if not file_only and isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)) and not focused.isReadOnly():
            focused.copy(); return
        if self.canvas.busy or self.canvas.transform is not None or self.canvas.layer_index < 0: return
        layer = self.canvas.layers[self.canvas.layer_index]
        if not layer['visible']:
            self.status.setText('숨겨진 레이어는 복사할 수 없습니다. 레이어를 표시해 주세요.')
            return
        try:
            source = layer['image']
            selection = self.canvas.selection
            if selection is None:
                copied = source.copy()
            else:
                # Selection is stored in preview coordinates; apply it to the
                # full-resolution selected layer without including other layers.
                mask = prepare_selection_mask(selection,self.canvas.canvas_size)
                bounds = mask.getbbox()
                if bounds is None:
                    self.status.setText('선택된 영역이 없습니다.')
                    return
                x,y = layer['x'],layer['y']
                x0,y0 = max(0,x,bounds[0]),max(0,y,bounds[1])
                x1,y1 = min(self.canvas.canvas_size[0],x+source.width,bounds[2]),min(self.canvas.canvas_size[1],y+source.height,bounds[3])
                if x1<=x0 or y1<=y0:
                    self.status.setText('선택한 레이어에 복사할 내용이 없습니다.')
                    return
                copied = source.crop((x0-x,y0-y,x1-x,y1-y)).convert('RGBA')
                copied.putalpha(ImageChops.multiply(copied.getchannel('A'),mask.crop((x0,y0,x1,y1))))
            if copied.getchannel('A').getbbox() is None:
                self.status.setText('선택한 영역에 보이는 픽셀이 없습니다.')
                return
            if file_only: copy_png_file_to_clipboard(copied)
            else: copy_rgba_to_clipboard(copied)
            self._local_clipboard_image = copied
            self._local_clipboard_name = layer['name']
            if sys.platform == 'win32':
                import ctypes
                self._local_clipboard_sequence = ctypes.windll.user32.GetClipboardSequenceNumber()
            else:
                self._local_clipboard_sequence = None
            if file_only:
                self.status.setText('PNG 파일을 복사했습니다. 다른 앱에서 Ctrl+V로 붙여넣으세요.')
            else:
                self.status.setText('투명 PNG로 복사했습니다. 카카오톡에서 Ctrl+V로 붙여넣으세요.' if copied.getchannel('A').getextrema()[0] < 255 else '이미지를 복사했습니다. Ctrl+V로 붙여넣으세요.')
        except (OSError, ValueError, MemoryError) as error:
            QMessageBox.critical(self, '복사 오류', str(error))
    def copy_png_file(self):
        self.copy_selection(file_only=True)
    def save_selected_layer_png(self):
        if self.canvas.busy or self.canvas.transform is not None or self.canvas.layer_index < 0: return
        layer=self.canvas.layers[self.canvas.layer_index]
        if not layer['visible']:
            QMessageBox.information(self,'숨겨진 레이어','선택한 레이어를 표시한 뒤 저장해 주세요.');return
        name=os.path.splitext(layer['name'])[0]+'.png'
        path,_=QFileDialog.getSaveFileName(self,'선택 레이어 PNG 저장',name,'PNG 이미지 (*.png)')
        if not path:return
        if not path.lower().endswith('.png'):path+='.png'
        try:
            clear_hidden_rgb(layer['image']).save(path,format='PNG')
            self.status.setText(f'선택 레이어 PNG 저장 완료: {path}')
        except (OSError,ValueError,MemoryError) as error:
            QMessageBox.critical(self,'PNG 저장 실패',str(error))
    def unique_copy_name(self, name):
        stem,ext=os.path.splitext(name)
        base=re.sub(r' \([0-9]+\)$','',stem) or stem
        existing={layer['name'] for layer in self.canvas.layers}
        index=1
        while f'{base} ({index}){ext}' in existing: index+=1
        return f'{base} ({index}){ext}'
    def paste_into_document(self, image, name, prepared=False, background=False):
        incoming = image if prepared else ImageOps.exif_transpose(image).convert('RGBA')
        if self.active_document < 0 or self.original is None:
            self.load_image(incoming, name)
            return
        width, height = self.canvas.canvas_size
        layers = [layer.copy() for layer in self.canvas.layers]
        layers.append({'name': name, 'image': incoming, 'x': (width-incoming.width)//2,
                       'y': (height-incoming.height)//2, 'visible': True})
        if background and (width*height >= 1_000_000 or incoming.width*incoming.height >= 1_000_000):
            self.worker=PasteCompositeWorker(layers,(width,height))
            self.worker.done.connect(lambda base,preview:self.finish_paste(layers,base,preview))
            self.worker.failed.connect(self.failed);self.worker.finished.connect(lambda:self.set_enabled(False))
            self.set_loading_message('이미지를 붙여넣고 있습니다');self.set_enabled(True)
            self.status.setText('이미지를 붙여넣고 있습니다…');self.worker.start();return
        self.canvas.snapshot();self.canvas.layers=layers;self.canvas.layer_index=len(layers)-1
        self.canvas.revision=next(_content_revisions);self.canvas.result=self.canvas.selection=None
        self.recompose_layers()
        self.status.setText('사진을 새 레이어로 붙여넣었습니다. Ctrl+T로 크기와 위치를 조절하세요.')
    def finish_paste(self, layers, base, preview):
        self.canvas.snapshot();self.canvas.layers=layers;self.canvas.layer_index=len(layers)-1
        self.canvas.revision=next(_content_revisions);self.canvas.result=self.canvas.selection=None
        self.original=self.canvas.original=base;self.canvas.image=preview;self.canvas.quick_lab=None
        self.canvas.marks=Image.new('L',preview.size,0);self.canvas.refresh();self.canvas.changed.emit()
        self.status.setText('사진을 새 레이어로 붙여넣었습니다. Ctrl+T로 크기와 위치를 조절하세요.')
    def paste_image(self):
        if self.canvas.busy: return
        clipboard = QApplication.clipboard()
        try:
            if sys.platform == 'win32' and self._local_clipboard_image is not None and self._local_clipboard_sequence is not None:
                import ctypes
                if ctypes.windll.user32.GetClipboardSequenceNumber() == self._local_clipboard_sequence:
                    self.paste_into_document(self._local_clipboard_image, self.unique_copy_name(self._local_clipboard_name), prepared=True, background=True)
                    return
            data = clipboard.mimeData()
            # File copies contain the original file, which can have more detail
            # than the clipboard's additional thumbnail representation.
            if data is not None and data.hasUrls():
                for url in data.urls():
                    if url.isLocalFile():
                        path = url.toLocalFile()
                        try:
                            with Image.open(path) as im:
                                self.paste_into_document(im, os.path.basename(path), background=True)
                            return
                        except (OSError, ValueError):
                            continue
            qimg = clipboard.image()
            if not qimg.isNull():
                rgba = qimg.convertToFormat(QImage.Format_RGBA8888)
                ptr = rgba.bits(); ptr.setsize(rgba.byteCount())
                image = Image.frombytes('RGBA', (rgba.width(), rgba.height()), bytes(ptr), 'raw', 'RGBA', rgba.bytesPerLine())
                self.paste_into_document(image, '붙여넣은 사진', background=True)
                return
            # Some Windows applications expose CF_DIB or copied file names
            # in a form not recognized as a Qt image.
            if sys.platform == 'win32':
                from PIL import ImageGrab
                item = ImageGrab.grabclipboard()
                if isinstance(item, Image.Image):
                    self.paste_into_document(item, '붙여넣은 사진', background=True)
                    return
                if isinstance(item, list):
                    for path in item:
                        try:
                            with Image.open(path) as im:
                                self.paste_into_document(im, os.path.basename(path), background=True)
                            return
                        except (OSError, ValueError, TypeError):
                            continue
            QMessageBox.information(self, '붙여넣을 사진 없음', '사진을 복사한 뒤 Ctrl+V를 눌러 주세요.')
        except Exception as e:
            QMessageBox.critical(self, '사진 붙여넣기 오류', str(e))
    def run(self, automatic, subjects='person'):
        if self.canvas.busy or self.canvas.image is None or (self.worker is not None and self.worker.isRunning()): return
        marks = np.asarray(self.canvas.marks).copy()
        if not automatic and not np.any(marks == 1):
            QMessageBox.information(self, '선택 영역 필요', '남길 피사체를 초록색 펜으로 먼저 칠해 주세요.'); return
        focus = self.canvas.selection.copy() if automatic and self.canvas.selection is not None else None
        image = self.canvas.image.copy()
        result_size = result_box = None
        subject_roi = None
        if automatic:
            index = self.canvas.layer_index
            if index < 0 or index >= len(self.canvas.layers): return
            layer = self.canvas.layers[index]
            if not layer['visible']:
                QMessageBox.information(self, '숨겨진 레이어', '선택한 레이어를 표시한 뒤 작업해 주세요.'); return
            width, height = self.canvas.canvas_size
            lw, lh = layer['image'].size; x, y = layer['x'], layer['y']
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + lw), min(height, y + lh)
            if x0 >= x1 or y0 >= y1:
                QMessageBox.information(self, '선택한 레이어', '선택한 레이어가 캔버스 밖에 있습니다.'); return
            pw, ph = self.canvas.image.size
            px0, py0 = min(pw-1, round(x0 * pw / width)), min(ph-1, round(y0 * ph / height))
            px1, py1 = round(x1 * pw / width), round(y1 * ph / height)
            px1, py1 = max(px0 + 1, min(pw, px1)), max(py0 + 1, min(ph, py1))
            result_size, result_box = (pw, ph), (px0, py0, px1, py1)
            image = layer['image'].crop((x0-x, y0-y, x1-x, y1-y)).convert('RGBA').resize((px1-px0, py1-py0), R.LANCZOS)
            if focus is not None:
                local_focus = focus.resize((pw, ph), R.NEAREST).crop(result_box)
                focus = local_focus if local_focus.getbbox() else None
                if subjects in ('all','objects'):
                    if focus is None:
                        self.status.setText('선택한 레이어 위에 선택 영역을 만들어 주세요.');return
                    subject_roi = focus.getbbox()
        self.worker = Worker(image, marks, automatic, roi=subject_roi, focus=focus, tolerance=self.tolerances['subject'], quality=self.quality_box.currentData(), result_size=result_size, result_box=result_box, subjects=subjects)
        self.worker.done.connect(self.finished); self.worker.failed.connect(self.failed)
        self.worker.finished.connect(lambda: self.set_enabled(False))
        self.set_loading_message('선택 영역을 분석하고 있습니다' if focus is not None else ('사물을 찾고 있습니다' if subjects=='objects' else '인물을 찾고 있습니다'))
        self.set_enabled(True)
        self.status.setText(('선택 범위에서 사람과 사물을 찾고 있습니다…' if subjects=='all' else ('선택한 레이어에서 사물을 찾고 있습니다…' if subjects=='objects' else ('선택 범위에서 사람을 찾고 있습니다…' if focus is not None else '사람을 찾고 가장자리를 세밀하게 다듬고 있습니다…'))) if automatic else '선택한 피사체의 경계를 계산하고 있습니다…')
        self.worker.start()
    def auto(self): self.run(True)
    def auto_objects(self): self.run(True, 'objects')
    def select(self): self.run(False)
    def finished(self, result):
        if not np.any(np.asarray(result) >= 128):
            self.status.setText('선택한 레이어에서 대상을 찾지 못했습니다. 범위를 넓히거나 선택 해제 후 다시 시도하세요.')
            return
        self.canvas.snapshot()
        self.canvas.selection = result
        self.canvas.refresh()
        self.status.setText('피사체의 점선 윤곽을 확인한 뒤 누끼따기를 누르세요.')
    def failed(self, error):
        self.status.setText('처리하지 못했습니다.')
        QMessageBox.critical(self, '처리 오류', error)
    def clear(self):
        self.canvas.snapshot()
        self.canvas.marks = Image.new('L', self.canvas.image.size, 0); self.canvas.refresh(); self.set_enabled()
        self.status.setText('펜 표시를 지웠습니다.')
    def reset(self):
        self.canvas.snapshot()
        self.canvas.result = None; self.canvas.selection = None; self.canvas.refresh(); self.set_enabled()
        self.status.setText('원본을 표시합니다. 펜 표시는 유지됩니다.')
    def export_image_data(self):
        if self.canvas.image is None or self.original is None:return None
        image=self.original.copy()
        if self.canvas.result is not None:
            mask=self.canvas.result.resize(image.size,R.LANCZOS)
            image.putalpha(ImageChops.multiply(image.getchannel('A'),mask))
        return clear_hidden_rgb(image)
    def write_export(self,path,settings):
        if settings.get('format')=='GIF' and settings.get('animate'):
            write_animated_gif(path,self.canvas.layers,self.canvas.canvas_size,settings,self.canvas.result)
            self.status.setText(f'내보내기 완료: {path}');return True
        image=self.export_image_data()
        if image is None:return False
        fmt=settings['format'];alpha=image.getchannel('A')
        if fmt=='JPEG' or not settings.get('transparency',True):
            background=Image.new('RGBA',image.size,'white');background.alpha_composite(image);image=background.convert('RGB')
        if fmt=='JPEG':image.save(path,'JPEG',quality=settings.get('quality',90))
        elif fmt=='GIF':
            rgb=image.convert('RGB');indexed=rgb.quantize(colors=255,method=Image.Quantize.MEDIANCUT)
            if settings.get('transparency',True):
                transparent=alpha.point(lambda value:255 if value<128 else 0)
                indexed.paste(255,mask=transparent);indexed.save(path,'GIF',transparency=255,disposal=2,optimize=False)
            else:indexed.save(path,'GIF')
        else:image.save(path,'PNG')
        self.status.setText(f'내보내기 완료: {path}');return True
    def choose_export_path(self,settings):
        if self.active_document<0 or self.original is None:return
        fmt=settings['format'];suffix={'PNG':'.png','JPEG':'.jpg','GIF':'.gif'}[fmt]
        name=os.path.splitext(self.documents[self.active_document]['name'])[0]+suffix
        path,_=QFileDialog.getSaveFileName(self,'내보내기',name,{'PNG':'PNG 이미지 (*.png)','JPEG':'JPG 이미지 (*.jpg *.jpeg)','GIF':'GIF 이미지 (*.gif)'}[fmt])
        if not path:return
        valid=('.jpg','.jpeg') if fmt=='JPEG' else (suffix,)
        if not path.lower().endswith(valid):path+=suffix
        if fmt=='GIF' and settings.get('animate'):
            layers=[layer.copy() for layer in self.canvas.layers]
            result=self.canvas.result.copy() if self.canvas.result is not None else None
            self.worker=GifExportWorker(path,layers,self.canvas.canvas_size,dict(settings),result)
            self.worker.done.connect(self.gif_export_done);self.worker.failed.connect(self.failed);self.worker.finished.connect(lambda:self.set_enabled(False))
            self.set_loading_message('GIF 애니메이션을 만들고 있습니다');self.set_enabled(True);self.status.setText('GIF 애니메이션을 만들고 있습니다…');self.worker.start();return
        try:self.write_export(path,settings)
        except (OSError,ValueError,MemoryError) as error:QMessageBox.critical(self,'내보내기 실패',str(error))
    def gif_export_done(self,path):
        self.status.setText(f'내보내기 완료: {path}')
    def quick_export_png(self):
        if not self.canvas.busy and self.canvas.transform is None:self.choose_export_path({'format':'PNG','transparency':True,'quality':90})
    def export_as_dialog(self):
        if self.canvas.busy or self.canvas.transform is not None or self.original is None:return
        dialog=ExportDialog(self.original,self.preferences.get('export',{}),self,layer_count=sum(1 for layer in self.canvas.layers if layer['visible']));self.apply_dialog_theme(dialog)
        if dialog.exec_()==QDialog.Accepted:self.choose_export_path(dialog.values())
    def export_preferences(self):
        if self.canvas.busy or self.canvas.transform is not None or self.original is None:return
        dialog=ExportDialog(self.original,self.preferences.get('export',{}),self,settings_only=True,layer_count=max(2,sum(1 for layer in self.canvas.layers if layer['visible'])));self.apply_dialog_theme(dialog)
        if dialog.exec_()!=QDialog.Accepted:return
        previous=self.preferences.get('export');self.preferences['export']=dialog.values()
        try:save_settings(self.preferences)
        except OSError as error:
            self.preferences['export']=previous;QMessageBox.critical(self,'설정 저장 오류',str(error));return
        self.status.setText('내보내기 기본 설정을 저장했습니다.')
    def export_ppt(self):
        if self.canvas.busy or self.canvas.transform is not None: return
        self.capture_document()
        available=[(i,doc['name']) for i,doc in enumerate(self.documents) if doc.get('original') is not None and doc.get('canvas_size')]
        if not available:
            QMessageBox.information(self,'내보낼 탭 없음','먼저 사진이나 새 문서를 열어 주세요.');return
        dialog=PptExportDialog(available,self)
        self.apply_dialog_theme(dialog)
        if dialog.exec_()!=QDialog.Accepted: return
        indices=dialog.selected_indices()
        if not indices: return
        default_name=suggested_pptx_filename(self.documents[indices[0]]['name'])
        path,_=QFileDialog.getSaveFileName(self,'PPT로 내보내기',default_name,'PowerPoint 프레젠테이션 (*.pptx)')
        if not path:return
        if not path.lower().endswith('.pptx'):path+='.pptx'
        snapshots=[]
        for index in indices:
            doc=self.documents[index]
            snapshots.append({'name':doc['name'], 'canvas_size':doc['canvas_size'],
                              'layers':[layer.copy() for layer in doc['layers']],
                              'result':doc.get('result')})
        self.worker=PptExportWorker(snapshots,dialog.slide_size.currentData(),path)
        self.worker.done.connect(self.ppt_export_done);self.worker.failed.connect(self.failed)
        self.worker.finished.connect(lambda:self.set_enabled(False))
        self.set_loading_message('PPT를 만들고 있습니다')
        self.set_enabled(True);self.status.setText('선택한 탭을 PPT 슬라이드로 내보내고 있습니다…');self.worker.start()
    def ppt_export_done(self,path):
        self.status.setText(f'PPT 내보내기 완료: {path}')
        QMessageBox.information(self,'PPT 내보내기 완료',f'PPT 파일을 저장했습니다.\n{path}')
    def save(self, checked=False, force_dialog=False):
        if self.canvas.busy or self.active_document < 0: return False
        doc = self.documents[self.active_document]
        path = doc['save_path']
        if force_dialog or not path:
            path, _ = QFileDialog.getSaveFileName(self, 'PNG 저장', os.path.splitext(doc['name'])[0] + '.png', 'PNG 이미지 (*.png)')
        if not path: return False
        if not path.lower().endswith('.png'): path += '.png'
        try:
            out = self.original.copy()
            a = np.asarray(out.getchannel('A'), dtype=np.uint16)
            if self.canvas.result is not None:
                b = np.asarray(self.canvas.result.resize(out.size, R.LANCZOS), dtype=np.uint16)
                out.putalpha(Image.fromarray((a*b//255).astype('uint8')))
            out = clear_hidden_rgb(out)
            dpi = doc.get('dpi')
            if isinstance(dpi, (tuple, list)) and len(dpi) == 2 and all(isinstance(v, (int, float)) and v > 0 for v in dpi):
                out.save(path, dpi=tuple(dpi))
            else:
                out.save(path)
            doc['save_path'] = path; doc['unsaved'] = False; doc['saved_signature'] = self.document_signature()
            doc['name'] = os.path.basename(path); self.tabs.setTabText(self.active_document, doc['name']); self.file_name.setText(doc['name'])
            self.status.setText(f'저장 완료: {path}')
            return True
        except Exception as e:
            QMessageBox.critical(self, '저장 오류', str(e)); return False
    def save_as(self):
        return self.save(force_dialog=True)
    def closeEvent(self, event):
        if self.canvas.transform is not None: self.cancel_transform()
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, '처리 중', '작업이 끝난 후 창을 닫아 주세요.'); event.ignore()
        else:
            while self.documents:
                if not self.close_tab(self.active_document): event.ignore(); return
            if self in Window.windows: Window.windows.remove(self)
            event.accept()

class DialogTitleIconFilter(QObject):
    def eventFilter(self, obj, event):
        if sys.platform == 'win32' and event.type() == QEvent.Show and isinstance(obj,QDialog):
            import ctypes
            from ctypes import wintypes
            user32=ctypes.windll.user32
            user32.GetWindowLongW.argtypes=[wintypes.HWND,ctypes.c_int]
            user32.SetWindowLongW.argtypes=[wintypes.HWND,ctypes.c_int,ctypes.c_long]
            user32.SetWindowPos.argtypes=[wintypes.HWND,wintypes.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.UINT]
            parent=obj.parentWidget()
            owner=parent.window() if parent is not None else None
            prefs=getattr(owner,'preferences',{})
            set_native_titlebar_theme(obj,prefs.get('theme','dark')=='dark')
            hwnd=int(obj.winId());style=user32.GetWindowLongW(hwnd,-20)
            user32.SetWindowLongW(hwnd,-20,style|0x00000001)
            user32.SetWindowPos(hwnd,None,0,0,0,0,0x0001|0x0002|0x0004|0x0020)
        return False

class TooltipDelayStyle(QProxyStyle):
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_ToolTip_WakeUpDelay: return 500
        if hint == QStyle.SH_ToolTip_FallAsleepDelay: return 0
        return super().styleHint(hint, option, widget, returnData)

if __name__ == '__main__':
    import faulthandler
    import traceback
    log_directory = os.path.dirname(SETTINGS_PATH) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(log_directory, exist_ok=True)
    log_path = os.path.join(log_directory, '오류기록.log')
    error_log = open(log_path, 'a', encoding='utf-8', buffering=1)
    faulthandler.enable(file=error_log, all_threads=True)
    def qt_message_handler(message_type, context, message):
        error_log.write(f'Qt [{int(message_type)}]: {message}\n')
        error_log.flush()
    qInstallMessageHandler(qt_message_handler)
    def report_exception(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb, file=error_log)
        error_log.flush()
        QMessageBox.critical(None, '오류 발생', '작업 중 오류가 발생했습니다. 오류기록.log를 확인해 주세요.\n\n' + str(exc))
    sys.excepthook = report_exception
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('MiniPhotoshop.Desktop.Editor')
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)),'assets','mini_photoshop.ico')))
    # Packaging smoke test exercises the same worker and real model as the UI.
    test_image = os.environ.get('MINI_PHOTOSHOP_TEST_IMAGE')
    test_output = os.environ.get('MINI_PHOTOSHOP_TEST_OUTPUT')
    if test_image and test_output:
        worker = Worker(Image.open(test_image).convert('RGBA'), None, True, quality='fast', subjects=os.environ.get('MINI_PHOTOSHOP_TEST_SUBJECTS','person'))
        def finish_test(message, code):
            with open(test_output, 'w', encoding='utf-8') as result_file:
                result_file.write(message)
            app.exit(code)
        worker.done.connect(lambda mask: finish_test('OK' if mask.getbbox() else 'NO_PERSON', 0 if mask.getbbox() else 1))
        worker.failed.connect(lambda message: finish_test(message, 1))
        worker.start()
        sys.exit(app.exec_())
    dialog_icon_filter=DialogTitleIconFilter(app);app.installEventFilter(dialog_icon_filter)
    app.setStyleSheet('QToolTip { color: #f4f7fc; background-color: #2a3342; border: 1px solid #566780; border-radius: 5px; padding: 7px 10px; font-size: 12px; }')
    app.setStyle(TooltipDelayStyle(app.style()))
    window = Window(); window.show()
    sys.exit(app.exec_())



