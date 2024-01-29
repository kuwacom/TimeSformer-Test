@REM venv をセットアップ
py -3.10 -m venv venv
.\venv\Scripts\activate

@REM モジュールの準備
pip install -r req.txt

@REM TimeSformerをclone
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose

@REM mmaction2をインストール
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
cd ../

@REM create model file
mkdir models