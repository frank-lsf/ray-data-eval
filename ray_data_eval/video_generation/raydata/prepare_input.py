files = [
    "youtube-8m-sample-sampled/high-res/WmQRXhDvC84_286_mp4_avc1.4D401F_1280x720_30.mp4",
    "youtube-8m-sample-sampled/high-res/e_XBNCzq8TQ_270_mp4_avc1.4D401F_1280x720_25.mp4",
    "youtube-8m-sample-sampled/high-res/yB94lxpywPk_170_mkv_avc1.64001F_1280x720_30.mkv",
    "youtube-8m-sample-sampled/high-res/ueVcs0u2Oy4_479_mkv_avc1.64001F_1280x720_30.mkv",
    "youtube-8m-sample-sampled/high-res/4zGf2vH4F-c_139_mp4_avc1.4D401F_1280x720_24.mp4",
    "youtube-8m-sample-sampled/low-res/Q7IB4v6JRxs_120_mp4_avc1.42001E_320x240_30.mp4",
    "youtube-8m-sample-sampled/low-res/JkjW4ye2sEI_378_mp4_avc1.42001E_320x240_30.mp4",
    "youtube-8m-sample-sampled/low-res/uetJ6xSeOZo_257_mp4_avc1.42001E_320x240_30.mp4",
    "youtube-8m-sample-sampled/low-res/iGWkVr9Np38_179_mp4_avc1.42001E_320x240_30.mp4",
    "youtube-8m-sample-sampled/low-res/0LqQ4DYZquE_336_mp4_avc1.42001E_320x240_30.mp4",
]

ret = []
for path in files:
    filename = path.split("/")[-1]
    n_frames = int(filename[12:15])
    for i in range(0, n_frames, 8):
        ret.append(f"{path}#{i}")

print(ret)
