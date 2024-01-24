# What Is This
これは TimeSformer をテストする際に利用できるツールセットです。

# How To Use
現在は Windows のみサポートしています。

## やり方1
この方法ではすべて自動で設定可能です。

動作はCPU限定となります。(torch類がCPU)

```
windows-setup.bat
```

をクリックすれば自動でセットアップが開始します。

## やり方2
この方法では反手動で設定可能です。

初めに

```
windows-env-setup.bat
```

上記のバッチを実行して、次に `req-cuda-12.1.txt` 等を利用して必要な python パッケージをインストールしていきます。

※cuda等のtorchを使う場合は各マシンに合わせてインストールしてください。

次に以下のバッチを実行して `TimeSformer` のセットアップをしてください。
```
windows-TimeSformer-setup.bat
```


# モジュールエラーについて

最新の torch バージョンでは絶対にバグが発生するため、以下の手順に沿ってファイルを変更してください。

<br>

https://github.com/facebookresearch/TimeSformer/issues/63

上記のを例に `TimeSformer/Timesformer/models/resnet_helper.py` 内の
```
from torch.nn.modules.linear import _LinearWithBias
```
をコメントアウトしてください。


https://github.com/NVIDIA/apex/issues/1048#issuecomment-877851575

次に、上記のを例に `TimeSformer/Timesformer/models/vit_utils.py` 内の
```
from torch._six import container_abcs
```
を
```
import collections.abc as container_abcs
```
に置き換えてください。