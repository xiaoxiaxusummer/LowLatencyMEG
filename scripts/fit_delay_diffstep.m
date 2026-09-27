
clear all; clc;

tomeratio=0.5;
merging_type = strcat('pr_sdxl_ddiff/tome_',num2str(tomeratio),'_perlayer/');
mkdir(merging_type);

fun_type = "linear"; % ["exp", "log", "poly2", "linear"]
delay_file_name = "delays_tome_0.5_cfg2-5";
energy_file_name = "energy_tome_0.5_cfg2-5";
flops_file_name = "flops_tome_0.5_cfg2-5";
load(strcat("/homes/xx623/demo/",merging_type, delay_file_name,".mat"), "diffusion_steps", "e2e_delays", "denoiser_delays", "unet_block_delays"); 
load(strcat("/homes/xx623/demo/",merging_type, flops_file_name,".mat"), "flops_text_encoder", "flops_unets", "flops_vae", "macs_text_encoder", "macs_unets", "macs_vae"); 
load(strcat("/homes/xx623/demo/",merging_type, energy_file_name,".mat"), "e2e_energy", "text_encoder_energy", "unet_block_energy", "decoder_energy"); 
param_dict = {};

% FLOPs单位全部由GFLOPs变为TeraFLOPs (TFLOPs)
flops_text_encoder = sum(flops_text_encoder,3)/10^3;
flops_unets = flops_unets/10^3;
flops_vae = flops_vae/10^3;

% MACs单位全部由GMACs变为TeraMACs (TMACs)
macs_text_encoder = sum(macs_text_encoder,3)/10^3;
macs_unets = macs_unets/10^3;
macs_vae = macs_vae/10^3;

% delay单位由second 到 millisecond 
e2e_delays = e2e_delays * 10^3;
denoiser_delays = denoiser_delays*10^3;
unet_block_delays = unet_block_delays*10^3;

% 能耗单位全部由 joule 变为 hectojoule (hJ)
e2e_energy = e2e_energy/10^2;
unet_block_energy = unet_block_energy/10^2;
text_encoder_energy = text_encoder_energy/10^2;
decoder_energy = decoder_energy/10^2;

diffusion_steps = cast(diffusion_steps,'double');
disp(diffusion_steps);

e2e_flops = flops_text_encoder+flops_vae+flops_unets;
flops_unets_single_step = mean(flops_unets./diffusion_steps); % TFLOPs/diffusion step 
flops_text_encoder = mean(flops_text_encoder);
flops_vae = mean(flops_vae);

% 拟合diffusion_steps和flops_unets的关系
[F3, x3, txt3, txt_real3] = fit_data(fun_type, diffusion_steps, flops_unets, 'Denoiser TFLOPs', 1);
% 拟合flops_text_encoder+flops_vae+flops_unets和e2e_delays的关系
% -------- Total model flops = flops_unets_single_step *num_diff + flops_text_encoder+ flops_vae ----------
[F4, x4, txt4, txt_real4] = fit_data(fun_type, diffusion_steps, e2e_flops, 'Model TFLOPs', 2);
disp(['Denoiser flops = ', num2str(x3(2)), '*num_diff + ', num2str(x3(1)), 'TFLOPs;']);
disp(['Text encoder flops = ', num2str(flops_text_encoder), ' TFLOPs']);
disp(['Decoder flops = ', num2str(flops_vae), ' TFLOPs']);
disp(['Total flops (assumed) = ', num2str(flops_unets_single_step), '*num_diff + ',  num2str(x3(1)+flops_text_encoder+flops_vae)]);
disp(['Total flops (from fitting) = ', num2str(x4(2)), '*num_diff + ', num2str(x4(1))]);
para_dict.flops_unets_single_step = flops_unets_single_step;
para_dict.flops_text_encoder = flops_text_encoder;
para_dict.flops_vae = flops_vae;
fig = figure; hold on; box on;
plot(diffusion_steps,F3(x3, diffusion_steps),'b-','linewidth',1.5);
plot(diffusion_steps,flops_unets,'ko');
plot(diffusion_steps,F4(x4, diffusion_steps),'r--','linewidth',1.5);
plot(diffusion_steps,e2e_flops,'k*'); 
xlabel("Number of diffusion steps", "Interpreter","latex");
ylabel("Model/denoiser TFLOPs");
legend([txt3, "Denoiser TFLOPs, measured", txt4, "Model TFLOPs, measured"], 'Interpreter', 'latex', 'location', 'NorthWest');
frame = getframe(fig); img = frame2im(frame); 
saveas(fig,strcat(merging_type,"/fit_",fun_type,"_",flops_file_name,".png"));
savefig(fig,strcat(merging_type,"/fit_",fun_type,"_",flops_file_name,".fig"));


% 拟合flops_unets和denoiser_delays的关系
[F1, x1, txt1, txt_real1] = fit_data(fun_type, flops_unets, denoiser_delays, 'Denoising delays', 1);
% ============== 拟合flops_text_encoder+flops_vae+flops_unets和e2e_delays的关系 =============
% -------- Computing delay =  computing_efficiency [second/TFLOPs] * TFLOPs + basic_delay ----------
[F2, x2, txt2, txt_real2] = fit_data(fun_type, e2e_flops, e2e_delays, 'Computing delays, ', 2);
disp(['Computing efficiency:', num2str(x1(2)), ' second/TFLOPs (denoiser); ', num2str(x2(2)), ' second/TFLOPs (overall); ']);
para_dict.computing_efficiency = x2(2);  % [second/TFLOPs]
para_dict.basic_delay = x2(1);
fig = figure; hold on; box on;
plot(flops_unets,F1(x1, flops_unets),'b-','linewidth',1.5);
plot(flops_unets,denoiser_delays,'ko'); 
plot(e2e_flops,F2(x2, e2e_flops),'r--','linewidth',1.5);
plot(e2e_flops,e2e_delays,'k*'); 
xlabel("Number of Terabit FLOPs", "Interpreter","latex");
ylabel("Computing/denoising delay (seconds)");
legend([txt1, "Denoising delay, measured", txt2, "Computing delay, measured"], 'Interpreter', 'latex', 'location', 'NorthWest');
frame = getframe(fig); img = frame2im(frame); 
saveas(fig,strcat(merging_type,"/fit_",fun_type,"_",delay_file_name,".png"));
savefig(fig,strcat(merging_type,"/fit_",fun_type,"_",delay_file_name,".fig"));


% ========= 拟合num_diff_step和denoiser_delays的关系 ==========
[F7, x7, txt7, txt_real7] = fit_data(fun_type, diffusion_steps, denoiser_delays, 'Denoising delays', 1);
% 拟合flops_text_encoder+flops_vae+flops_unets和e2e_delays的关系
% -------- Computing delay =  computing_efficiency [second/TFLOPs] * TFLOPs + basic_delay ----------
[F8, x8, txt8, txt_real8] = fit_data(fun_type, diffusion_steps, e2e_delays, 'Computing delays', 2);
fig = figure; hold on; box on;
plot(diffusion_steps,F7(x7, diffusion_steps),'b-','linewidth',1.5);
plot(diffusion_steps,denoiser_delays,'ko'); 
plot(diffusion_steps,F8(x8, diffusion_steps),'r--s','linewidth',1.5);
plot(diffusion_steps,e2e_delays,'k*'); 
%% 验证一下
e2e_delay_model = para_dict.computing_efficiency * (para_dict.flops_unets_single_step * diffusion_steps + para_dict.flops_text_encoder + para_dict.flops_vae) + para_dict.basic_delay; 
plot(diffusion_steps, e2e_delay_model, 'g-.', 'linewidth', 1.5);
xlabel("Number of diffusion steps", "Interpreter","latex");
ylabel("Computing/denoising delay (seconds)");
legend([txt7, "Denoising delay, measured", txt8, "Computing delay, measured", "The utilized delay model"], 'Interpreter', 'latex', 'location', 'NorthWest');
frame = getframe(fig); img = frame2im(frame); 
saveas(fig,strcat(merging_type,"/fit_",fun_type,"_diff_",delay_file_name,".png"));
savefig(fig,strcat(merging_type,"/fit_",fun_type,"_diff_",delay_file_name,".fig"));



% 拟合flops_unets和unet_block_energy的关系
[F5, x5, txt5, txt_real5] = fit_data(fun_type, flops_unets, unet_block_energy, 'Denoiser energy consumption', 1);
% 拟合flops_text_encoder+flops_vae+flops_unets和e2e_energy的关系, 即
% energy_efficiency*(flops_unets/n)*n + energy_efficiency*(flops_encoder+flops_decoder) + bias
% =energy_efficiency*(flops_unets+flops_encoder+flops_decoder) + bias
e2e_flops = flops_text_encoder+flops_vae+flops_unets;
e2e_energy_data = unet_block_energy + text_encoder_energy + text_encoder_energy;
[F6, x6, txt6, txt_real6] = fit_data(fun_type, e2e_flops, e2e_energy, 'Model energy consumption', 2);
disp(['TFLOPs per hJ:', num2str(x5(2)), 'hJ/TFLOPs (denoiser); ', num2str(x6(2)), 'hJ/TFLOPs (overall); ']);

fig = figure; hold on; box on;
plot(flops_unets,F5(x5, flops_unets),'b-','linewidth',1.5);
plot(flops_unets,unet_block_energy,'ko'); 
plot(e2e_flops,F6(x6, e2e_flops),'r--','linewidth',1.5);
plot(e2e_flops,e2e_energy,'k*'); 
xlabel("Number of Terabit FLOPs", "Interpreter","latex");
ylabel("Model/denoiser energy consumption (hJ)");
legend([txt5, "Denoiser energy consumption, measured", txt6, "Model energy consumption, measured"], 'Interpreter', 'latex', 'location', 'NorthWest');
frame = getframe(fig); img = frame2im(frame); 
saveas(fig,strcat(merging_type,"/fit_",fun_type,"_",energy_file_name,".png"));
savefig(fig,strcat(merging_type,"/fit_",fun_type,"_",energy_file_name,".fig"));

para_dict.energy_efficiency = x6(2); % hJ/TFLOPs
para_dict.bias_energy = x6(1); % hJ


% clear fig frame img;
save(strcat(merging_type,"/fit_",fun_type,"_",delay_file_name,".mat"),"para_dict");




function [F, x, txt, txt_real] = fit_data(fun_type, xdata, y, txt, ind, ydata)
    if strcmp(fun_type,"exp")
        F = @(x,xdata)x(1)*exp(x(2)*xdata) - x(3)*xdata; % + x(3)*exp(-x(4)*xdata);
        x0 = [1 1 2];
        % % txt = strcat(num2str(x(1)), "exp(", num2str(x(2)), "x)-",num2str(x(3)), "x");
        txt = strcat(txt, ", $a_", num2str(ind), "\exp(\lambda_",num2str(ind),"x)-b_",num2str(ind),"x$");
        [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,xdata,y);
        txt_real = string([num2str(x(1)), 'exp(', num2str(x(2)), 'x)-', num2str(x(3)), 'x']);
    elseif strcmp(fun_type,"log")
        F = @(x,xdata)-x(3)*log(x(1)*xdata+x(2))+x(4)*xdata;
        x0 = [1 0.2 1 1];
        txt = strcat(txt, ",$b_",num2str(ind), "\log(a_",num2str(ind), " x)-\lambda_",num2str(ind)," x$");
        [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,xdata,y);
        txt_real = string(['-', num2str(x(3)), 'log(', num2str(x(1)), 'x+', num2str(x(2)), ')+' num2str(x(4)), 'x']);
    elseif strcmp(fun_type,"linear")
        F = @(x,xdata)x(2)*xdata+x(1);
        x0 = [1 0];
        txt = strcat(txt, ",$a_",num2str(ind), "x+b_",num2str(ind),"$");
        [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,xdata,y);
        txt_real = string([num2str(x(2)), 'x+', num2str(x(1))]);
    elseif strcmp(fun_type,"linear_nobias")
        F = @(x,xdata)x*xdata;
        x0 = 1;
        txt = strcat(txt, ",$a_",num2str(ind), "x$");
        [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,xdata,y);
        txt_real = string([num2str(x), 'x']);
    elseif strcmp(fun_type,"poly2")
        F = @(x,xdata)x(3)*xdata.^2+x(2)*xdata+x(1);
        x0 = [1 0 0];
        txt = strcat(txt, ",$a_",num2str(ind), "x^2+b_",num2str(ind),"x+c_",num2str(ind),"$");
        [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,xdata,y);
        txt_real = string([num2str(x(3)), 'x^2+', num2str(x(2)), 'x+', num2str(x(1))]);
    elseif strcmp(fun_type,"bilinear")
        F = @(x,xdata)x(2)*xdata(1,:)+x(1)*xdata(2,:);
        x0 = [2 0];
        lb = [0.01, 0]; ub = [Inf, Inf];
        txt = strcat(txt, ",$a_",num2str(ind), "x_1+b_",num2str(ind),"x_2+c_",num2str(ind),"$");
        [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,xdata,y,lb,ub);
        txt_real = string([num2str(x(2)), 'x_1+', num2str(x(2)), 'x_2']);
    end
    x = round(x,4);
end