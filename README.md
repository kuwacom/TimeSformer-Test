# What Is This
これは TimeSformer をテストする際に利用できるツールセットです。

# How To Use
公式のドキュメントをもとに環境を構築していきます
https://mmaction2.readthedocs.io/en/latest/get_started/installation.html

## Python環境の作成
### 仮想環境の作成
`venv`を利用して環境を構築していきます

それぞれのOSにあったコマンドを実行してください
(コマンド等が違う場合はその都度修正してください)

> Windows
```shell
py -3.10 -m venv venv
```
> Linux
```bash
python3 -m venv venv
```

### モジュールのインストール
次にモジュールをインストールします

先ほど準備した仮想環境へアクティベートした状態で行ってください

```
./venv/Scripts/activate
```
#### Python モジュールのインストール
最初に`Python`のモジュールのインストールをしていきます

プリセットでは二つの環境のみ用意してありますのでそれ以外の場合は各自変更してください

> CUDA 12.1
```bash
pip install -r req-cuda-12.1.txt
```
torch類は https://pytorch.org/get-started/locally/ より手動で入れます
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
> CPU
```bash
pip install -r req.txt
```

#### OpenMMLab のライブラリのインストール
次に動作に必要な`OpenMMLab`のライブラリをインストールしていきます

```bash
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose
```

### MMAction2 をインストール
次に`MMAction2`本体をインストールしていきます

**こちらも先ほど準備した仮想環境へアクティベートした状態で行ってください**

以下のコマンドは本プロジェクトの`root`にて行ってください
```bash
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```

以上でセットアップは終了です
