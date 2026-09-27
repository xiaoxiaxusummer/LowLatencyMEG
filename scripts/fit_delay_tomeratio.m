
clear all; clc;

is_merging_cross = false;
if is_merging_cross
    merging_type = "pr_sdxl_tome_attn_perlayer_cross/";
else 
    merging_type = "pr_sdxl_tome_attn_perlayer/";
end
mkdir(merging_type);

fun_type = "exp"; % ["exp", "log"]
data_file_name = "delays_step_6_cfg2-5";

data = load(strcat("/homes/xx623/demo/",merging_type, data_file_name,".mat"), ...
                    "tomeratios", "e2e_delays", "denoiser_delays", ...
                    "transformers_delays","unet_block_delays"); 
transformers_delays = data.transformers_delays;
tomeratios = data.tomeratios;
e2e_delays = data.e2e_delays;
denoiser_delays = data.denoiser_delays;
unet_block_delays = data.unet_block_delays;

[F1, x_denoising, txt1, txt_real1] = fit_data(fun_type, tomeratios, denoiser_delays, "Denoising delay", 1);
[F2, x_e2e, txt2, txt_real2] = fit_data(fun_type, tomeratios, e2e_delays, "Computing delay", 2);


data_file_name = "flops_step_6_cfg2-5";
flops_data = load(strcat("/homes/xx623/demo/",merging_type, data_file_name,".mat"), ...
    'flops_text_encoder', 'macs_text_encoder', 'params_text_encoder', 'flops_uncon_text_encoder', 'macs_uncon_text_encoder', ...
    'params_uncon_text_encoder', "flops_unets",'macs_unets', 'params_unets'); 

flops_unets_sum = sum(flops_data.flops_unets,2).'/10^4;
[F3, x_flops, txt3, txt_real3] = fit_data(fun_type, tomeratios, flops_unets_sum, "$10\times$TFLOPs for Denoiser", 3);

% unet_delays =  sum(unet_block_delays,2).'/10^3/6;
% [F3, x_unet, txt3, txt_real3] = fit_data(fun_type, tomeratios, unet_delays, "UNet delay");

save(strcat(merging_type,"/fit_",fun_type,"_",data_file_name,".mat"));

fig = figure; hold on; box on;
plot(tomeratios,F1(x_denoising,tomeratios),'b-','linewidth',1.5,'MarkerSize',8);
plot(tomeratios,denoiser_delays,'ko','MarkerSize',8); 
plot(tomeratios,F2(x_e2e,tomeratios),'r--','linewidth',1.5,'MarkerSize',8);
plot(tomeratios,e2e_delays,'k*','MarkerSize',8); 
plot(tomeratios,F3(x_flops,tomeratios),'k--','linewidth',1.5,'MarkerSize',8);
plot(tomeratios,flops_unets_sum,'k>','MarkerSize',8); 
xlabel("Merging ratio $x$", "Interpreter","latex");
ylabel("Delay (seconds) and $10\times$TFLOPs for diffusion models", "Interpreter","latex");
legend([txt1, "Denoising delay, raw data", txt2, "Computing delay, raw data",txt3, "$10\times$TFLOPs for Denoiser, raw data"], 'Interpreter', 'latex', 'location', 'East');
% frame = getframe(fig); img = frame2im(frame);
% imwrite(img,strcat("fit_",fun_type,"_",data_file_name,".png"));
saveas(fig,strcat(merging_type,"/fit_",fun_type,"_",data_file_name,".png"));
savefig(fig,strcat(merging_type,"/fit_",fun_type,"_",data_file_name,".fig"));


[F4, x_flops_delay, txt4, txt_real4] = fit_data('linear', flops_unets_sum*10, denoiser_delays, "Denoising delay", 1);
fig = figure; hold on; box on;
plot(flops_unets_sum*10,F4(x_flops_delay,flops_unets_sum*10),'b-','linewidth',1.5,'MarkerSize',8);
plot(flops_unets_sum*10,denoiser_delays,'ko','MarkerSize',8); 
xlabel("TFLOPs of denoising diffusion models", "Interpreter","latex");
ylabel("Delay (seconds)", "Interpreter","latex");
legend([txt4, "Denoising delay, raw data"], 'Interpreter', 'latex');
saveas(fig,strcat(merging_type,"/fit_delayvsflops_",fun_type,"_",data_file_name,".png"));
savefig(fig,strcat(merging_type,"/fit_delayvsflops_",fun_type,"_",data_file_name,".fig"));


% legend_txt = [];
for i = 1:size(transformers_delays,2)
    % legend_txt = [legend_txt; string(['transformers_',num2str(i)])];
    [F, x_tformer, txt, txt_real] = fit_data(fun_type, tomeratios, transformers_delays(:,i).', "Transformers delay");
    fig2 = figure; hold on; box on;
    plot(tomeratios,F(x_tformer,tomeratios),'r--','linewidth',1.5);
    plot(tomeratios, transformers_delays(:,i),'ko');
    legend([txt_real,"Transformers delay, raw data"]);
    xlabel("Merging ratio $x$", "Interpreter","latex");
    ylabel("Transformer delay (ms)");
    saveas(fig2,strcat(merging_type,"/transformers", num2str(i), "_",data_file_name,".png"));
end





function [F, x, txt, txt_real] = fit_data(fun_type, xdata, y, txt, ind)
    if strcmp(fun_type,"exp")
        F = @(x,xdata)x(1)*exp(x(2)*xdata) - x(3)*xdata; % + x(3)*exp(-x(4)*xdata);
        x0 = [1 1 2];
        % % txt = strcat(num2str(x(1)), "exp(", num2str(x(2)), "x)-",num2str(x(3)), "x");
        txt = strcat(txt, ", $a_", num2str(ind), "\exp(\lambda_",num2str(ind),"x)-b_",num2str(ind),"x$");
        [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,xdata,y)
        x = round(x,3);
        txt_real = string([num2str(x(1)), 'exp(', num2str(x(2)), 'x)-', num2str(x(3)), 'x']);
    elseif strcmp(fun_type,"log")
        F = @(x,xdata)-x(3)*log(x(1)*xdata+x(2))+x(4)*xdata;
        x0 = [1 0.2 1 1];
        txt = strcat(txt, ",$b_",num2str(ind), "\log(a_",num2str(ind), " x)-\lambda_",num2str(ind)," x$");
        [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,xdata,y)
        x = round(x,3);
        txt_real = string(['-', num2str(x(3)), 'log(', num2str(x(1)), 'x+', num2str(x(2)), ')+' num2str(x(4)), 'x']);
    elseif strcmp(fun_type,"linear")
        F = @(x,xdata)x(2)*xdata+x(1);
        x0 = [1 0];
        txt = strcat(txt, ",$a_",num2str(ind), "x+b_",num2str(ind),"$");
        [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,xdata,y)
        x = round(x,3);
        txt_real = string([num2str(x(2)), 'x+', num2str(x(1))]);
    end
end
