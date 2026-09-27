from calflops import calculate_flops

"""
FLOPs: [GFLOPs]
MACs: [GMACs]
parameters: [Billions]

"""
def FLOPs_text_encoder(tokenizer, text_encoder, inputs):

    if inputs["input_ids"].shape[1] < tokenizer.model_max_length:
        apply_num = tokenizer.model_max_length- inputs["input_ids"].shape[1]
        inputs["input_ids"].extend([0]*apply_num)
        # inputs["token_type_ids"].extend([0]*apply_num)
        inputs["attention_mask"].extend([0]*apply_num)

    inputs.to(text_encoder.device)

    flops, macs, params = calculate_flops(model=text_encoder, kwargs = inputs, print_results=False, output_as_string = False)

    flops, macs, params = flops/10**9, macs/10**9, params/10**9
    print("Text Encoder FLOPs:%s GFLOPs  MACs:%s GMACs  Params:%s B \n" %(flops, macs, params))

    return flops, macs, params             


def FLOPs_unet(unet, args, kwargs):
   # args 记录顺序输入，kwargs记录键值输入， 由于原始unet使用混合输入方式，因此两者都要输入，否则在hook中无法捕捉
   flops, macs, params = calculate_flops(model=unet, 
                                      args = args, 
                                      kwargs=kwargs,
                                      print_results=False, 
                                      output_as_string = False,
                                      )
   flops, macs, params = flops/10**9, macs/10**9, params/10**9
   print("UNet FLOPs:%s GFLOPs  MACs:%s GMACs  Params:%s B \n" %(flops, macs, params))

   
   return flops, macs, params

def FLOPs_vae_decode(decoder, args, kwargs):
    flops, macs, params = calculate_flops(model=decoder, args = args,  kwargs=kwargs,  print_results=False,   output_as_string = False,)
    flops, macs, params = flops/10**9, macs/10**9, params/10**9
    print("VAE FLOPs:%s GFLOPs  MACs:%s GMACs  Params:%s B \n" %(flops, macs, params))
    return flops, macs, params