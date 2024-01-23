@REM TimeSformerをclone
git clone https://github.com/facebookresearch/TimeSformer
cd TimeSformer
py setup.py build develop
cd ../

@REM create model file
mkdir models