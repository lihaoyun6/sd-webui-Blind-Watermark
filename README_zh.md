# SD Blind Watermark
将不可见的隐形自定义水印插入到你的图片中. 

## 截图
<img src="./images/ui.jpg"/>  

## 使用说明
此插件允许你将一张64*64像素的水印图像以肉眼不可见的方式插入你的图片中 (支持手动插入或在图像生成后自动插入)  

在首次使用前请前往`Blind Watermark`>`Get watermark data`标签页, 并将你准备好的水印图像拖入图像框. 再把转换出的数据文本复制粘贴到`Settings`>`BlindWatermark`>`Watermark data`并保存.  

如果启用了`Automatically embed the watermark after generation`选项, 建议同时启用 `Make SD Blind Watermark run after any other extensions`, 以防止其他插件对图像处理时破坏掉水印信息.  

每张图片都会使用随机生成的Passcode进行加密, **请勿将图像的Passcode告诉任何人, 它是从图像中读取水印的唯一凭据!!!**  

## 安装
1. 前往 SD WebUI 的 `扩展` 标签页
2. 点击 `从网址安装` 子标签
3. 将 `https://github.com/lihaoyun6/sd-webui-Blind-Watermark` 粘贴进网址输入框
4. 点击 `安装` 并等待完成
5. 提示安装成功后重载 WebUI 即可启用

## 鸣谢
- [BlindWatermark](https://github.com/fire-keeper/BlindWatermark) @darksouls4  
- [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) @AUTOMATIC1111  
