[app]

# 应用标题
title = PK10预测系统

# 包名
package.name = pk10predictor

# 包域名
package.domain = org.pk10

# 源代码目录
source.dir = .

# 源代码包含的文件
source.include_exts = py,png,jpg,kv,atlas,json

# 应用版本
version = 1.0

# 应用需求（Python包）- 移除torch和torchvision
requirements = python3,kivy,numpy,scipy,requests,certifi

# 禁用SDL2_image的额外格式支持
p4a.bootstrap = sdl2

# 支持的架构
android.archs = arm64-v8a,armeabi-v7a

# Android权限
android.permissions = INTERNET,ACCESS_NETWORK_STATE,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# Android API版本
android.api = 31
android.minapi = 21
android.ndk = 25b
android.skip_update = False
android.accept_sdk_license = True

# 方向
orientation = portrait

# 全屏
fullscreen = 0

# Android入口点
android.entrypoint = org.kivy.android.PythonActivity

# Android应用主题
android.apptheme = @android:style/Theme.NoTitleBar

# 日志级别
log_level = 2

# 警告级别
warn_on_root = 1


[buildozer]

# 日志级别
log_level = 2

# 警告
warn_on_root = 1
