# テストの前に以下を実行してください
# mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest ./mmaction2

import os
import json
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from mmaction.apis import inference_recognizer, init_recognizer

def resize_frame(frame, width=None, height=None):
    """
    フレームのサイズを変更する関数。

    Args:
        frame (numpy.ndarray): サイズを変更するフレーム（画像）。
        width (int, optional): 変更後の幅。デフォルトは None。
        height (int, optional): 変更後の高さ。デフォルトは None。

    Returns:
        numpy.ndarray: サイズを変更したフレーム。
    """
    if width is None and height is None:
        return frame

    if width is None:
        aspect_ratio = height / float(frame.shape[0])
        width = int(frame.shape[1] * aspect_ratio)
    elif height is None:
        aspect_ratio = width / float(frame.shape[1])
        height = int(frame.shape[0] * aspect_ratio)

    resized_frame = cv2.resize(frame, (width, height))

    return resized_frame

def get_resized_dimensions(frame, width=None, height=None):
    """
    フレームをリサイズした後のサイズを取得する関数。

    Args:
        frame (numpy.ndarray): サイズを変更するフレーム（画像）。
        width (int, optional): 変更後の幅。デフォルトは None。
        height (int, optional): 変更後の高さ。デフォルトは None。

    Returns:
        tuple: 変更後の幅と高さのタプル。
    """
    if width is None and height is None:
        return frame.shape[1], frame.shape[0]

    if width is None:
        aspect_ratio = height / float(frame.shape[0])
        width = int(frame.shape[1] * aspect_ratio)
    elif height is None:
        aspect_ratio = width / float(frame.shape[1])
        height = int(frame.shape[0] * aspect_ratio)

    return width, height


def insert_text_on_frame(
        frame, text, org=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.4,
        text_color=(255, 255, 255), outline_color=(0, 0, 0), outline_size=2,
        thickness=2, line_spacing=30
        ):
    """
    フレームにテキストを挿入し、黒でアウトラインを付ける

    Args:
        frame (numpy.ndarray): テキストを挿入するフレーム（画像）
        text (str): 挿入するテキスト
        org (tuple, optional): テキストを挿入する位置 (x, y)。デフォルトは (10, 30)
        font (int, optional): 使用するフォント。デフォルトは cv2.FONT_HERSHEY_SIMPLEX
        font_scale (float, optional): フォントのスケール。デフォルトは 0.4
        text_color (tuple, optional): テキストの色 (B, G, R)。デフォルトは (255, 255, 255)
        outline_color (tuple, optional): アウトラインの色 (B, G, R)。デフォルトは (0, 0, 0)
        thickness (int, optional): テキストとアウトラインの太さ。デフォルトは 2
        line_spacing (int, optional): 行間のスペース。デフォルトは 30

    Returns:
        numpy.ndarray: テキストを挿入したフレーム
    """
    # テキストを改行で分割
    lines = text.split('\n')

    # 改行ごとにテキストを挿入
    y = org[1]
    for line in lines:
        # アウトラインを描画
        cv2.putText(frame, line, (org[0], y), font, font_scale, outline_color, thickness + outline_size, cv2.LINE_AA)
        # テキストを描画
        cv2.putText(frame, line, (org[0], y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += int(font_scale * line_spacing)

    return frame

def process_segment(video_path, model, test_pipeline, temp_path, label_path, start_frame, end_frame, fps, segment_duration, frame_height, processing_frame_height, total_frames):
    cap = cv2.VideoCapture(video_path)

    # セグメントのフレームを一時ファイルとして保存
    segment_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        segment_frames.append(frame)

    # セグメントを一時ファイルに書き出す
    segment_filename = f'{temp_path}/segment_{start_frame}_{end_frame}.mp4'
    out = cv2.VideoWriter(segment_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, get_resized_dimensions(frame, None, processing_frame_height))
    for frame in segment_frames:
        out.write(resize_frame(frame, None, processing_frame_height))
    out.release()


    # 推論結果を取得してテキストを挿入した動画を生成
    pred_result = inference_recognizer(model, segment_filename, test_pipeline)
    
    # 推論結果を整形
    pred_scores = pred_result.pred_score.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=lambda x: x[1], reverse=True)
    top5_labels = score_sorted[:5]

    labels = open(label_path).readlines()
    labels = [x.strip() for x in labels]
    top5_results = [(labels[k[0]], k[1]) for k in top5_labels]
    results = [(labels[k[0]], k[1]) for k in score_sorted]

    added_text = []
    results_object = {}
    for idx, frame in enumerate(segment_frames):
        display_text = 'kuwacom/TimeSformer-Tools\n'
        display_text += f'Segment Interval: {segment_duration}s\n'
        display_text += f'Segment Frame: {start_frame} ~ {end_frame}\n'  # セグメントの範囲を表示
        display_text += f'Frame: {start_frame + idx}/{total_frames}\n'

        results_object['segmentStartFrame'] = start_frame
        results_object['segmentEndFrame'] = end_frame
        results_object['labels'] = results
        

        # リザルトを改行していい感じにする
        for idx, result in enumerate(top5_results):
            display_text += f'{idx+1}. {result[0]}: {result[1]:.3f}\n'
        
        # フレームにテキストを挿入
        frame_with_text = insert_text_on_frame(frame.copy(), display_text, font_scale=(frame_height / 1000))
        added_text.append(frame_with_text)
        
    print(f'Done Processing : {start_frame} ~ {end_frame}')
    return added_text, results_object

def split_segment_detect(
        video_path, model, label_path, test_pipeline=None,
        processing_frame_height=None, segment_duration=10,
        output_path='./', output_video_filename='output.mp4', output_labeldata_filename='label-data.json',
        temp_path='./temp', thread=2
        ):
    """
    動画をセグメントに分割し、各セグメントを推論モデルで処理して結果を描画し、結合して出力

    Args:
        video_path (str): 入力動画ファイルのパス。処理対象となる動画ファイルのパスを指定します。
        model (nn.Module): 推論に使用するモデル。動画セグメントを処理するためのニューラルネットワークモデルです。
        label_path (str): ラベルマップのファイルパス。モデルが出力するクラスラベルと対応するラベル名が書かれたファイルのパスを指定します。
        test_pipeline (Optional[Compose]): テスト用の前処理パイプライン。動画フレームに対して行う前処理のパイプラインオブジェクトを指定します。デフォルトはNoneです。
        processing_frame_height (Optional[int]): 処理時のフレームの高さ。動画フレームを処理する際にリサイズする高さを指定します。デフォルトはNoneで、オリジナルのサイズが使用されます。
        segment_duration (int): セグメントの長さ（秒単位）。動画を分割する際のセグメントの長さを秒単位で指定します。デフォルトは10秒です。
        output_path (str): 出力ファイルのディレクトリパス。処理後の出力動画およびラベルデータファイルを保存するディレクトリのパスを指定します。デフォルトは現在のディレクトリ('./')です。
        output_video_filename (str): 出力動画ファイルの名前。処理後の動画ファイルの名前を指定します。デフォルトは 'output.mp4' です。
        output_labeldata_filename (str): 出力ラベルデータファイルの名前。処理結果のラベルデータを保存するファイルの名前を指定します。デフォルトは 'label-data.json' です。
        temp_path (str): 一時ファイルを保存するディレクトリのパス。処理中に使用する一時ファイルを保存するディレクトリのパスを指定します。デフォルトは './temp' です。
        thread (int): スレッド数。処理に使用するスレッドの数を指定します。デフォルトは2です。

    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if processing_frame_height is None:
        processing_frame_height = frame_height

    # segment_duration 秒間のフレーム数を計算
    frames_per_segment = int(fps * segment_duration)
    print(f'Total Frame: {total_frames}')

    os.makedirs(temp_path, exist_ok=True)
    
    processed_videos = []
    processed_results = []
    frame_num = 0
    with ThreadPoolExecutor(max_workers=thread) as executor: # threadの制限かける
        futures = []
        while frame_num < total_frames:
            
            start_frame = frame_num # セグメントの最初のフレーム番号を設定
            end_frame = min(start_frame + frames_per_segment, total_frames)  # セグメントの終了フレーム番号
            print(f'NOW Processing Frame:{start_frame} ~ {end_frame}')

            frame_num += (end_frame - start_frame)

            # スレッドを作成
            future = executor.submit(process_segment, video_path, model, test_pipeline, temp_path, label_path, start_frame, end_frame, fps, segment_duration, frame_height, processing_frame_height, total_frames)
            futures.append(future)

            if len(futures) == thread: # 制限分のthreadに達したらいったん処理待ち
                for future in futures:
                    processed_video, processed_result = future.result()
                    processed_videos.append(processed_video)
                    processed_results.append(processed_result)
                futures = []

        # 残りの未処理セグメントを処理
        for future in futures:
            processed_video, processed_result = future.result()
            processed_videos.append(processed_video)
            processed_results.append(processed_result)
    cap.release()

    # 処理済み動画を結合して出力
    save_video(os.path.join(output_path, output_video_filename), concat_videos(processed_videos), fps, (frame_width, frame_height))
    with open(os.path.join(output_path, output_labeldata_filename), 'w') as json_file:
        json.dump({
            'segmentInterval': segment_duration,
            'labelData': processed_results
        }, json_file, indent=4)

    print(f'処理済みの動画を {os.path.join(output_path, output_video_filename)} に書き出しました！')
    print(f'ラベルデータを {os.path.join(output_path, output_video_filename)} に書き出しました！')

def concat_videos(video_segments):
    return [item for sublist in video_segments for item in sublist]
    
def save_video(filename, frames, fps, frame_size):
    frame_width, frame_height = frame_size
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()


if __name__ == '__main__':
    # MMAction2の設定ファイルとチェックポイントファイル
    config_file = './mmaction2/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
    checkpoint_file = './mmaction2/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'

    # MMAction2のモデルを初期化
    model = init_recognizer(config_file, checkpoint_file, device='cuda:0')

    enable_fp16 = True
    cuda_float = 32
    if torch.cuda.is_available() & enable_fp16:
        model.half() # fp16
        cuda_float = 16

    # 処理する動画ファイルとラベルファイルのパス
    video_path = './mmaction2/demo/demo.mp4'
    label_path = './mmaction2/tools/data/kinetics/label_map_k400.txt'

    segment_duration = 10 # 分割するフレームの間隔(秒)

    output_path = './'
    output_video_filename = f'video-s{segment_duration}-fp{cuda_float}.mp4'
    output_labeldata_filename = f'label-s{segment_duration}-fp{cuda_float}.json'

    # 動画の処理とプレビューの表示を実行
    split_segment_detect(
        video_path, model, label_path, segment_duration=segment_duration,
        output_path=output_path, output_video_filename=output_video_filename, output_labeldata_filename=output_labeldata_filename,
        processing_frame_height=None,
        thread=8
        )
