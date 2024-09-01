import matplotlib.pyplot as plt
import numpy as np
normal_t_block = [0.0035472, 0.0035472, 0.0036288, 0.00369696, 0.00354144, 0.00382048, 0.00426944, 0.00515744, 0.0066848, 0.009344, 0.0143699, 0.0246717, 0.0448416, 0.0861843, 0.166051, ]
const_t_block = [0.00361888, 0.00343552, 0.0034704, 0.00353056, 0.00358176, 0.00369344, 0.00425184, 0.00511104, 0.00655936, 0.00900608, 0.0138176, 0.0234032, 0.0424394, 0.0822515, 0.159638, ]

normal_t_warp = [0.00362528, 0.00357696, 0.00362272, 0.00376512, 0.0035968, 0.00392448, 0.004552, 0.00537664, 0.00695168, 0.00969312, 0.0150016, 0.0256163, 0.0467069, 0.089081, 0.168478, ]
const_t_warp = [0.00353728, 0.00345344, 0.00343552, 0.00351904, 0.0035312, 0.00358976, 0.00427712, 0.00513984, 0.00667648, 0.00915712, 0.0142144, 0.0243814, 0.0440208, 0.0846003, 0.164252, ]

normal_t_thread = [0.00354848, 0.00359104, 0.00352672, 0.00365344, 0.00369344, 0.00389472, 0.00428832, 0.0052976, 0.00682304, 0.00952832, 0.0142842, 0.0240902, 0.0434557, 0.0835594, 0.161427, ]
const_t_thread = [0.00552384, 0.00479232, 0.00480288, 0.00486112, 0.00462912, 0.00462048, 0.00477792, 0.00527648, 0.0072192, 0.00992927, 0.0146186, 0.0244845, 0.0431523, 0.082287, 0.159271, ]

normal_t_rand = [0.00396192, 0.00402432, 0.00400448, 0.0040864, 0.00413664, 0.00453376, 0.0056256, 0.00650304, 0.00818304, 0.0109072, 0.0161437, 0.0263271, 0.0466717, 0.0874979, 0.167563, ]
const_t_rand = [0.0238771, 0.0196022, 0.0174429, 0.0166029, 0.0158797, 0.0125734, 0.011929, 0.0146566, 0.0240109, 0.0431718, 0.080713, 0.151548, 0.295819, 0.584119, 1.15969, ]

timings = [2**x for x in range(10, 25)]

ratio_block = np.array(const_t_block) / np.array(normal_t_block)
ratio_warp = np.array(const_t_warp) / np.array(normal_t_warp)
ratio_thread = np.array(const_t_thread) / np.array(normal_t_thread)
ratio_rand = np.array(const_t_rand) / np.array(normal_t_rand)

def create_graph(ratio, title, filename):
    plt.figure(figsize=(10, 6))
    plt.semilogx(timings, ratio, marker='o')
    plt.title(title)
    plt.xlabel('Data Length')
    plt.ylabel('Ratio (Const/Normal)')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

create_graph(ratio_block, 'Ratio of Const/Normal Times (Block)', 'ratio_block.png')
create_graph(ratio_warp, 'Ratio of Const/Normal Times (Warp)', 'ratio_warp.png')
create_graph(ratio_thread, 'Ratio of Const/Normal Times (Thread)', 'ratio_thread.png')
create_graph(ratio_rand, 'Ratio of Const/Normal Times (Random)', 'ratio_rand.png')

