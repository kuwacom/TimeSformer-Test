@REM venv をセットアップ
py -3.10 -m venv venv
.\venv-3.10\Scripts\activate

@REM モジュールの準備
pip install -r req.txt

@REM TimeSformerをclone
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose

@REM create model file
mkdir models