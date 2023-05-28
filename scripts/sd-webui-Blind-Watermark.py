import gradio as gr
import numpy as np 
import random, os, cv2, base58

from PIL import Image
from io import BytesIO
from modules.shared import opts
from modules.BlindWatermark import watermark
from modules.BlindWatermark.tools import cv_imwrite
from modules.images import save_image, get_next_sequence_number

from modules import scripts, script_callbacks, shared

extf, ext = os.path.split(scripts.basedir())
prefix = 'zzzz'
err = "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"

def base58_to_image(base58_str: str) -> Image.Image:
	byte_data = base58.b58decode(base58_str)
	image_data = BytesIO(byte_data)
	img = Image.open(image_data)
	return np.asarray(img)

def image_to_base58(image) -> str:
	if image == None:
		return ""
	if image.size != (64, 64):
		return "Watermark size must be 64*64px!"
	buffered = BytesIO()
	image.save(buffered, format="PNG")
	img_str = base58.b58encode(buffered.getvalue())
	return img_str.decode('utf-8')

err = base58_to_image(err)

def calc_res(img):
	if img is not None:
		im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		return [im.shape[1],im.shape[0]]
	return ["",""]

def emb_wm(img, passcode='', filename='', save=True):
	if img is None:
		return  [err], "Please select the image to be processed first!"
	if opts.data.get("sdbwm_wm_image", None) == None:
		print("Set watermark data in \"Settings\" > \"BlindWatermark\" first!")
		return [err], "Set watermark data in \"Settings\" > \"BlindWatermark\" first!"
	if passcode == '':
		passcode = get_pass()
	seed_a, seed_b, strength, shape, deep = de_pass(passcode)
	bwm = watermark(seed_a, seed_b, strength, block_shape=(shape,shape))
	bwm.read_ori_img(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
	wm_img = base58_to_image(opts.data.get("sdbwm_wm_image", None))
	result, info = bwm.read_wm(cv2.cvtColor(wm_img, cv2.COLOR_RGB2BGR))
	if not result:
		return [err], info
	output_img = bwm.embed()
	#embed_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
	if filename == '':
		filename = passcode
	#out_image=Image.fromarray(np.uint8(embed_img))
	fullfn, txt_fullfn = save_image(Image.fromarray(err), opts.data.get("sdbwm_output_path", "outputs/marked-images"), "", seed=filename)
	cv_imwrite(fullfn, output_img)
	out_image = Image.open(fullfn)
	if not save:
		os.remove(fullfn) 
	return out_image, fullfn

def emb_wm_ui(img, files, batch, same_pass):
	passkey = get_pass()
	if batch:
		imgs = []
		outpath = opts.data.get("sdbwm_output_path", "outputs/marked-images")
		for i in files:
			img = np.asarray(Image.open(BytesIO(i)))
			if same_pass:
				out_image, fullfn = emb_wm(img,passcode=passkey)
			else:
				out_image, fullfn = emb_wm(img)
			imgs.append(out_image)
		if same_pass:
			return passkey, imgs, f"Done! All images have been saved to {outpath}"
		return "", imgs, f"Done! All images have been saved to {outpath}"
	else:
		out_image, fullfn = emb_wm(img,passcode=passkey)
		return passkey, [out_image], fullfn

def ext_wm(img, passcode, channel, channel_invert, width, height):
	if "Y" not in channel and "U" not in channel and "V" not in channel:
		return passcode, "One of the Y/U/V channels is required!", [err]
	if img is None:
		return passcode, "Please select the image to be processed first!", [err]
	
	seed_a = seed_b = strength = shape = deep = 1
	try:
		seed_a, seed_b, strength, shape, deep = de_pass(passcode)
	except:
		return passcode, "Incorrect password!", [err]
	
	bwm = watermark(seed_a, seed_b, strength, wm_shape=(64,64), block_shape=(shape,shape))
	im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	size = (int(width), int(height))
	if (int(im.shape[1]), int(im.shape[0])) != size:
		im = cv2.resize(im, size)
	wm_imgs = bwm.extract(im, channel="".join(channel), invert="".join(channel_invert))
	out_images = list(map(lambda x: Image.fromarray(np.uint8(x)), wm_imgs))
	return passcode, "", out_images

def get_pass(s='', d=''):
	seed_a = random.randrange(1000,9999)
	seed_b = random.randrange(1000,9999)
	strength = opts.data.get("sdbwm_wm_strength", 15)
	shape = s or opts.data.get("tsdbwm_block_shape", 4)
	deep = d or opts.data.get("tsdbwm_dwt_deep", 1)
	bwm_str = str(seed_a)+str(seed_b)+str(strength)+str(shape)+str(deep)
	#bwm_str = str(int(bwm_str) * int(opts.data.get("sdbwm_main_key", "0")))
	#if len(bwm_str)%2 != 0:
	#	bwm_str += "0"
	bwm_pass = base58.b58encode(int(bwm_str).to_bytes(int(len(bwm_str)/2), 'big')).decode('utf-8')
	return bwm_pass

def de_pass(passcode):
	pass_bytes = base58.b58decode(passcode)
	passcode = str(int.from_bytes(pass_bytes, 'big'))
	#passcode = str(int(int.from_bytes(pass_bytes, 'big') / int(opts.data.get("sdbwm_main_key", "0"))))
	#if len(passcode) != 12:
	#	pass_t = str(int.from_bytes(pass_bytes, 'big'))[:-1]
	#	passcode = str(int(int(pass_t) / int(opts.data.get("sdbwm_main_key", "0"))))
	return int(passcode[0:4]), int(passcode[4:8]), int(passcode[8:10]), int(passcode[10]), int(passcode[11])

def show_batch(visible):
	return [{"visible": not visible, "__type__": "update"},{"visible": visible, "__type__": "update"},{"visible": visible, "__type__": "update"}]

def get_force():
	if ext.startswith(prefix):
		return True
	return False

def make_last():
	if get_force():
		os.rename(os.path.join(extf, ext), os.path.join(extf, ext.replace(prefix, '')))
	else:
		os.rename(os.path.join(extf, ext), os.path.join(extf,prefix+ext))
	shared.state.interrupt()
	shared.state.need_restart = True

def on_ui_tabs():
	with gr.Blocks(analytics_enabled=False) as sd_blind_watermark:
		with gr.Tabs(elem_id="images_history_tab") as tabs:
			with gr.TabItem('Embed', id='SDBWM_embed', elem_id="SDBWM_embed_tab"):
				with gr.Row():
					with gr.Column():
						input_img = gr.Image(label="Input Image", elem_id="SDBWM_input_image", source="upload", interactive=True, type="numpy", image_mode="RGBA")
						input_files = gr.File(label="Input Files", elem_id="SDBWM_input_files", file_types=['image'], file_count="multiple", type="binary", interactive=True, visible=False)
						with gr.Row():
							batch = gr.Checkbox(label="Batch Process", value=False, interactive=True)
							same_pass = gr.Checkbox(label="Use same passcode", value=False, interactive=True, visible=False)
						embed_button = gr.Button(value="Embed",variant="primary")
					with gr.Column(variant="panel"):
						#output_img = gr.Image(label="Output Image", elem_id="SDBWM_output_image", source="upload", interactive=False, type="numpy", image_mode="RGBA").style(height=500)
						output_img = gr.Gallery(label="Output Image", elem_id="SDBWM_output_image", show_label=False).style(columns=4, object_fit="scale-down")
						infotext = gr.Markdown(value="")
						wm_pass = gr.Textbox(label="Passocode",value="", interactive=False)
						
				batch.change(
					fn=lambda x: show_batch(x),
					inputs=[batch],
					outputs=[input_img, input_files, same_pass],
				)
				
				embed_button.click(
					fn=emb_wm_ui,
					inputs=[input_img, input_files, batch, same_pass],
					outputs=[wm_pass, output_img, infotext],
				)
				
				#input_img.change(
				#	fn=lambda: "",
				#	inputs=[],
				#	outputs=[infotext],
				#)
			
			with gr.TabItem('Extract', id='SDBWM_extract', elem_id="SDBWM_extract_tab"):
				with gr.Row():
					with gr.Column():
						input_imge = gr.Image(label="Input Image", elem_id="SDBWM_input_imagee", source="upload", interactive=True, type="numpy", image_mode="RGBA")
						with gr.Row(variant="panel"):
							with gr.Column():
								width = gr.Textbox(label="Resize width to", value="", placeholder="width", interactive=True)
								height = gr.Textbox(label="Resize height to", value="", placeholder="height", interactive=True)
							with gr.Column():
								channel_wm = gr.CheckboxGroup(["Y","U","V"],value=["Y","U","V"], label="Channels:", interactive=True)
								channel_invert = gr.CheckboxGroup(["Y","U","V"],value=None, label="Invert channels:", interactive=True)
						wm_passe = gr.Textbox(label="Passocode", show_label=False, value="", placeholder="Passocode", interactive=True)
						wrrtext = gr.Markdown(value="")
						extra_button = gr.Button(value="Extract",variant="primary")
					with gr.Column(variant="panel"):
						output_wm = gr.Gallery(label="Watermark", elem_id="SDBWM_output_wm", show_label=False).style(columns=4, rows=1,object_fit="scale-down")
						infotext = gr.Markdown(value="Please make sure that the input image has the same size as the original image. If the image is scaled, cropped or rotated, restore it to its original size and angle before extracting the watermark.")
						#output_wm = gr.Image(label="Watermark", elem_id="SDBWM_output_wm", source="upload", interactive=False, type="numpy", image_mode="RGBA")
				
				input_imge.change(
					fn=calc_res,
					inputs=[input_imge],
					outputs=[width, height],
				)
				
				extra_button.click(
					fn=ext_wm,
					inputs=[input_imge, wm_passe, channel_wm, channel_invert, width, height],
					outputs=[wm_passe, wrrtext, output_wm],
				)
				
				#channel_wm.change(
				#	fn=ext_wm,
				#	inputs=[input_imge, wm_passe, channel_wm, channel_invert],
				#	outputs=[wm_passe, wrrtext, output_wm],
				#)

				#channel_invert.change(
				#	fn=ext_wm,
				#	inputs=[input_imge, wm_passe, channel_wm, channel_invert],
				#	outputs=[wm_passe, wrrtext, output_wm],
				#)

			with gr.TabItem('Get watermark data', id='SDBWM_settings', elem_id="SDBWM_settings_tab"):
				with gr.Row():
					with gr.Column():
						wm_img = gr.Image(label="Watermark image", source="upload", interactive=True, type="pil", image_mode="L", elem_id="SDBWM_wmimage")
					with gr.Column(variant="panel"):
						gr.Markdown(value="Drag the watermark image into the image box and you will get the base58 data here.")
						gr.Markdown(value="Copy and paste it into \"Settings\" > \"BlindWatermark\" > \"Watermark data...\"")
						wm_base58 = gr.Textbox(show_label=False)
				wm_img.change(
					fn=image_to_base58,
					inputs=[wm_img],
					outputs=[wm_base58],
				)
					
	return (sd_blind_watermark , "Blind Watermark", "sd_blind_watermark"),

def create_settings_items():
	section = ('sd_blind_watermark_setting', 'BlindWatermark')
	#shared.opts.add_option('sdbwm_main_key', shared.OptionInfo("0", 'Master Key (Numbers only. Big = anti-exhaustion, but keys will get longer) [If you set this value, do not change it rashly! If you forget this key, all watermarks embedded with this key will not be able to be extracted!]', section=section))
	shared.opts.add_option('sdbwm_wm_image', shared.OptionInfo(
		"", 'Watermark data (goto tab "Blind Watermark" > "Get watermark data" to get data string and paste it here)', section=section
	))
	shared.opts.add_option('sdbwm_output_path', shared.OptionInfo(
		"outputs/marked-images", 'Output directory for watermarked images', section=section
	))
	#shared.opts.add_option("tsdbwm_wm_shape", shared.OptionInfo("64x64", "Size of the watermark image", gr.Radio, {"choices": ["32x32", "64x64"]}, section=section))
	opts.add_option("sdbwm_wm_strength", shared.OptionInfo(
		15, "Watermark strength (larger = more robust, but also more artifacts)",
		gr.Slider, {"minimum": 10, "maximum": 99, "step": 1}, section=section
	))
	shared.opts.add_option("tsdbwm_block_shape", shared.OptionInfo(
		"6", "Pixel block size (larger = more invisible, but image may not have enough space for watermark)",
		gr.Radio, {"choices": ["2", "4", "6", "8"]}, section=section
	))
	shared.opts.add_option("tsdbwm_dwt_deep", shared.OptionInfo(
		"1", "DWT depth (larger = more robust, but image may not have enough space for watermark)",
		gr.Radio, {"choices": ["1", "2", "3"]}, section=section
	))
	shared.opts.add_option('sdbwm_auto_wm', shared.OptionInfo(
		False, 'Automatically embed the watermark after generation (make sure this is the last plugin, otherwise the watermark will be overwritten)', section=section
	))
	shared.opts.add_option('sdbwm_directly_save', shared.OptionInfo(
		False, 'Embed the watermark directly into the original image generated by SD and save it instead of saving as a copy (passcode will be written to the pnginfo)', section=section
	))
	
class SDBWM(scripts.Script):
	def title(self):
		return 'Blind Watermark'
	
	def describe(self):
		return "Embed custom blind watermark into your images."
	
	def show(self, is_img2img):
		return scripts.AlwaysVisible
	
	
	def ui(self, is_img2img):
		self.force_last = gr.Checkbox(label='Make SD Blind Watermark run after any other extensions (will reload WebUI)', value=get_force(), interactive=True, elem_id="sdbwm_force_last")
		setattr(self.force_last,"do_not_save_to_config",True)
		
		self.force_last.change(
			fn=make_last,
			_js='restart_reload',
			inputs=[],
			outputs=[],
		)

	def process(self, p):
		pass
	
	def postprocess_image(self, p, pp):
		if opts.data.get("sdbwm_auto_wm", False):
			self.bwm_pass = get_pass()
			if pp.image.size <= (512, 512):
				self.bwm_pass = get_pass(s=4,d=1)
			print("Embedding watermark...")
			image = np.array(pp.image)
			if opts.data.get("sdbwm_directly_save", False):
				out_image, fullfn = emb_wm(image, passcode=self.bwm_pass, save=False)
				p.extra_generation_params["BWM pass"] = self.bwm_pass
				pp.image = out_image
			else:
				out_image, fullfn = emb_wm(image, passcode=self.bwm_pass)
				print(f"Saved: {fullfn}")

script_callbacks.on_ui_tabs(on_ui_tabs)
scripts.script_callbacks.on_ui_settings(create_settings_items)