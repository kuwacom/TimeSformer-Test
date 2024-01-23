import torch
import torchvision.transforms as transforms
import cv2
from timm.models.vision_transformer import TimeSformer

# # TimeSformerモデルのロード
# model = TimeSformer(img_size=224, num_classes=1000, num_frames=8, attention_type='divided_space_time')

# # モデルを評価モードに設定
# model.eval()



# モデルの設定
model_name = 'timesformer_divST_8x32_224'
num_classes = 1000  # クラス数は事前学習済みモデルのものに合わせる
num_frames = 8  # フレーム数

# TimeSformerモデルの作成
model = TimeSformer(
    img_size=224,
    num_classes=num_classes,
    num_frames=num_frames,
    attention_type='divided_space_time',
)

# 事前学習済みの重みのパス
pretrained_weights_path = './models/TimeSformer_divST_8x32_224_K400.pyth'

# 保存済みの重みを読み込む
checkpoint = torch.load(pretrained_weights_path, map_location='cpu')

# モデルに重みをロード
model.load_state_dict(checkpoint['model'])

# モデルを評価モードに設定
model.eval()




# カメラの設定
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラを指定

# フレームをリアルタイムで取得
while True:
    ret, frame = cap.read()

    # フレームが正常に取得されたら処理を行う
    if ret:
        # 画像の前処理（リサイズ、正規化など）
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_frame = transform(frame).unsqueeze(0)

        # 推論
        with torch.no_grad():
            output = model(input_frame)

        # ここでoutputを使用して必要な処理を行う（例：結果の表示、何かしらのアクションの実行）

        # ウィンドウにフレームを表示
        cv2.imshow('Real-time Inference', frame)

        # 'q'を入力するとループを終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# キャプチャを解放
cap.release()
cv2.destroyAllWindows()
