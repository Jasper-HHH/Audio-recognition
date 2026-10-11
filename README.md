# Audio-recognition

离线识别本地曲库中的歌曲，并输出对应的简谱文本。程序不训练模型、不需要 GPU，也不会访问网络。它用音频 chroma 特征做片段匹配；简谱来自曲库中预先准备的文本文件，不会从任意音频自动转谱。

## 环境

需要 Python 3.10 或更高版本。安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 准备曲库

每首参考曲需要一份音频和一份同名简谱文件：

```text
library/
	月亮代表我的心.wav
	月亮代表我的心.jianpu.txt
```

音频支持 WAV、FLAC、OGG、AIFF；MP3 是否可读取决于 `soundfile` 所用的 libsndfile 构建。若 MP3 无法读取，请先转换成 WAV。简谱文件使用 UTF-8 纯文本，内容会在识别命中后原样输出。

## 运行

```bash
python music_recognizer.py --input ./recording.wav --library ./library
```

曲库可以包含子目录。程序默认匹配分数阈值为 `0.62`，可用 `--threshold 0.68` 调高阈值以减少误报。匹配分数不是概率；录音版本、伴奏、速度或调性与参考音频差别较大时，识别率会下降。短于 5 秒的片段会被拒绝。

要识别曲库之外的歌曲，需要先添加该歌曲的参考音频和对应简谱。离线且没有参考曲库时，无法可靠判断任意歌曲的名字；从混音音频直接生成准确简谱也不在本工具的能力范围内。