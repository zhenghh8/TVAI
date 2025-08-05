import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize(attn_map, title, cmap='viridis', save_dir=None, vmin=0, vmax=1, fontsize=46): 
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(attn_map, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xticks(ticks=[0.5, 5.5, 10.5, 15.5, 20.5, 25.5, 30.5], labels=[0, 5, 10, 15, 20, 25, 30], fontsize=fontsize)
    plt.xlabel('Head', fontsize=fontsize)
    plt.yticks(ticks=[0.5, 5.5, 10.5, 15.5, 20.5, 25.5, 30.5], labels=[0, 5, 10, 15, 20, 25, 30], fontsize=fontsize)
    plt.ylabel('Layer', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    if save_dir is not None:
        plt.savefig(save_dir, transparent=True)
        plt.close()

def attn_sorted(attn_map, dim=-1): 
    sorted_indices = np.argsort(-attn_map, axis=dim)
    attn_map = np.take_along_axis(attn_map, sorted_indices, axis=dim)

    return attn_map

def combine_nonhall_attn(discriminative_nonhall_npy, generative_nonhall_npy, fig_title, save_path='./'):
    discriminative_nonhall = np.load(discriminative_nonhall_npy)
    generative_nonhall = np.load(generative_nonhall_npy)
    
    if discriminative_nonhall.shape != generative_nonhall.shape:
        raise ValueError(f"two npy file with different shape: {discriminative_nonhall.shape} vs {generative_nonhall.shape}")
    
    # mean
    combined_arr = (discriminative_nonhall + generative_nonhall) / 2.0
    visualize(combined_arr, title=fig_title, save_dir=save_path, vmax=0.6)


if __name__ == "__main__":
    # # nonhall combination
    # discriminative_nonhall_npy = "/home/zhenghh8/hallucination/TVAI/results_analysis/pope_coco/llava-1.5/pope_eval_with_attn_random_tokens_512_nonhall_img.npy"
    # generative_nonhall_npy = "/home/zhenghh8/hallucination/TVAI/results_analysis/chair/llava-1.5/chair_eval_with_attn_tokens_512.jsonl_nonhall_img_mean.npy"
    # combine_nonhall_attn(discriminative_nonhall_npy, generative_nonhall_npy, fig_title="VAR", save_path='./nonhall_img_mean.png')


    # discriminative_nonhall_npy = "/home/zhenghh8/hallucination/TVAI/results_analysis/pope_coco/llava-1.5/pope_eval_with_attn_random_tokens_512_nonhall_text.npy"
    # generative_nonhall_npy = "/home/zhenghh8/hallucination/TVAI/results_analysis/chair/llava-1.5/chair_eval_with_attn_tokens_512.jsonl_nonhall_text_mean.npy"
    # combine_nonhall_attn(discriminative_nonhall_npy, generative_nonhall_npy, fig_title="TAR", save_path='./nonhall_text_mean.png')


    # # single show
    # npy_file = "/home/zhenghh8/hallucination/TVAI/results_analysis/chair/llava-1.5/chair_eval_with_attn_tokens_512.jsonl_hall_img_mean.npy"
    # # npy_file = "/home/zhenghh8/hallucination/TVAI/results_analysis/pope_coco/llava-1.5/pope_eval_with_attn_random_tokens_512_hall_text.npy"
    # visualize(np.load(npy_file), title="VAR", save_dir='./generative_hall_img.png', vmax=0.6)

    pass