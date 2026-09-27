import numpy as np
import matplotlib.pyplot as plt


tomeratios = np.linspace(start=0, stop=0.975, num=40, endpoint=True).round(3)
delays = np.load("demo/pr_sdxl_tome/delays.npz")
delays = {'tomeratios': tomeratios, 'e2e_delays': delays["e2e_delays"], 'denoiser_delays': delays["denoiser_delays"]}
plt.plot(tomeratios, delays["e2e_delays"],label='computational delay')
plt.plot(tomeratios, delays["denoiser_delays"],label='denoising delay')
plt.xlabel("Merging ratio")
plt.ylabel("Delay (second)")
plt.legend()
plt.savefig("demo/pr_sdxl_tome/delays_ratio.png")

plt.clf()


import scipy.io as scio
scio.savemat('demo/pr_sdxl_tome/delays.mat', delays)