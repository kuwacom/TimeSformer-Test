@REM venv をセットアップ
py -3.10 -m venv venv
.\venv-3.10\Scripts\activate

@REM モジュールの準備
pip install -r req.txt

@REM TimeSformerをclone
git clone https://github.com/facebookresearch/TimeSformer
cd TimeSformer
py setup.py build develop
cd ../

@REM create model file
mkdir models