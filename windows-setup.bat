@REM venv をセットアップ
py -3.7 -m venv venv-3.7
.\venv-3.7\Scripts\activate

@REM モジュールの準備
pip install -r req.txt

@REM TimeSformerをclone
git clone https://github.com/facebookresearch/TimeSformer
cd TimeSformer
py setup.py build develop
cd ../

@REM create model file
mkdir models