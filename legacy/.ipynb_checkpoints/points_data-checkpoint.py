"""
Data file containing all the 2D and 3D points of every camera
"""

import json

import numpy as np

with open("legacy/real_worls_points.json") as f:
    data = json.load(f)

# Latest 10 of complete row (from 10th to 20th)
distorted_img_ch1_bottom_fances_top = [
    (112, 839),
    (157, 859),
    (207, 879),
    (288, 903),
    (461, 948),
    (767, 998),
    (1367, 1033),
    (1896, 1004),
    (2231, 952),
    (2378, 915),
    (2462, 889),
]
obj_ch1_bottom_fances_top = data["bottom_fances_top"][-11:]
assert len(distorted_img_ch1_bottom_fances_top) == len(obj_ch1_bottom_fances_top)

# Latest 10 of complete row (one less compared to previous because latest fance does not have top bar)
distorted_img_ch1_bottom_fances_bar_top = [
    (109, 793),
    (148, 800),
    (200, 810),
    (285, 823),
    (467, 844),
    (776, 869),
    (1371, 887),
    (1895, 876),
    (2221, 852),
    (2371, 834),
]
obj_ch1_bottom_fances_bar_top = data["bottom_fances_bar_top"][-10:]
assert len(distorted_img_ch1_bottom_fances_bar_top) == len(
    obj_ch1_bottom_fances_bar_top
)

distorted_img_ch1_cell_bottom = [
    (141, 704),
    (165, 655),
    (179, 650),
    (285, 624),
    (317, 616),
    (356, 609),
    (429, 663),
    (554, 653),
    (777, 550),
    (898, 539),
    (1036, 530),
    (1183, 524),
    (1339, 522),
    (1495, 526),
    (1643, 532),
    (1927, 645),
]
obj_ch1_cell_bottom = []
obj_ch1_cell_bottom.extend(data["bottom_cells_first_set_bottom"][:3])
obj_ch1_cell_bottom.extend(data["bottom_cells_first_set_bottom"][7:10])
obj_ch1_cell_bottom.append(data["bottom_cells_first_set_bottom"][-1])
obj_ch1_cell_bottom.append(data["bottom_cells_second_set_bottom"][0])
obj_ch1_cell_bottom.extend(data["bottom_cells_second_set_bottom"][2:9])
obj_ch1_cell_bottom.append(data["bottom_cells_second_set_bottom"][-1])
assert len(distorted_img_ch1_cell_bottom) == len(obj_ch1_cell_bottom)

distorted_img_ch1_cell_top = [
    (122, 654),
    (140, 649),
    (152, 643),
    (166, 637),
    (183, 631),
    (200, 624),
    (220, 616),
    (246, 610),
    (275, 604),
    (310, 592),
    (348, 582),
    (386, 573),
    (510, 548),
    (611, 528),
    (717, 512),
    (846, 500),
    (993, 487),
    (1155, 477),
    (1339, 474),
    (1513, 481),
    (1679, 491),
    (1830, 500),
    (1968, 524),
]
obj_ch1_cell_top = []
obj_ch1_cell_top.extend(data["bottom_cells_first_set_top"])
obj_ch1_cell_top.extend(data["bottom_cells_second_set_top"])
assert len(distorted_img_ch1_cell_top) == len(obj_ch1_cell_top)

distorted_img_ch1_middle_cell_bottom = [
    (430, 419),
    (463, 409),
    (499, 398),
    (540, 385),
    (821, 326),
    (894, 318),
    (974, 307),
    (1058, 300),
    (1148, 297),
    (1525, 297),
    (1614, 302),
    (1868, 370),
]
obj_ch1_middle_cell_bottom = []
obj_ch1_middle_cell_bottom.extend(data["middle_cells_bottom"][-21:-17])
obj_ch1_middle_cell_bottom.extend(data["middle_cells_bottom"][-13:-8])
obj_ch1_middle_cell_bottom.extend(data["middle_cells_bottom"][-5:-3])
obj_ch1_middle_cell_bottom.append(data["middle_cells_bottom"][-1])
assert len(distorted_img_ch1_middle_cell_bottom) == len(obj_ch1_middle_cell_bottom)

distorted_img_ch1_middle_cell_top = [
    (400, 405),
    (434, 393),
    (467, 379),
    (508, 367),
    (552, 353),
    (603, 339),
    (721, 310),
    (788, 299),
    (864, 287),
    (948, 277),
    (1036, 267),
    (1134, 264),
    (1234, 260),
    (1335, 259),
    (1437, 263),
    (1538, 264),
    (1632, 269),
    (1724, 279),
    (1891, 305),
]
obj_ch1_middle_cell_top = []
obj_ch1_middle_cell_top.extend(data["middle_cells_top"][-21:-15])
obj_ch1_middle_cell_top.extend(data["middle_cells_top"][-14:-2])
obj_ch1_middle_cell_top.append(data["middle_cells_top"][-1])
assert len(distorted_img_ch1_middle_cell_top) == len(obj_ch1_middle_cell_top)

ch1_img = []
ch1_img.extend(distorted_img_ch1_bottom_fances_top)
ch1_img.extend(distorted_img_ch1_bottom_fances_bar_top)
ch1_img.extend(distorted_img_ch1_cell_bottom)
# ch1_img.extend(distorted_img_ch1_cell_bottom[3:])
ch1_img.extend(distorted_img_ch1_cell_top)
# ch1_img.extend(distorted_img_ch1_cell_top[7:])
ch1_img.extend(distorted_img_ch1_middle_cell_bottom)
ch1_img.extend(distorted_img_ch1_middle_cell_top)
ch1_img = np.array(ch1_img, dtype=np.float32)

ch1_obj = []
ch1_obj.extend(obj_ch1_bottom_fances_top)
ch1_obj.extend(obj_ch1_bottom_fances_bar_top)
# ch1_obj.extend(obj_ch1_cell_bottom[3:])
ch1_obj.extend(obj_ch1_cell_bottom)
# ch1_obj.extend(obj_ch1_cell_top[7:])
ch1_obj.extend(obj_ch1_cell_top)
ch1_obj.extend(obj_ch1_middle_cell_bottom)
ch1_obj.extend(obj_ch1_middle_cell_top)
ch1_obj = np.array(ch1_obj, dtype=np.float32)

# np.save("ch1_obj.npy", ch1_obj)
# np.save("ch1_img.npy", ch1_img)

distorted_img_ch4_bottom_fances_top = [
    (136, 844),
    (161, 859),
    (206, 887),
    (263, 915),
    (380, 965),
    (565, 1026),
    (1032, 1105),
    (1617, 1128),
    (2093, 1086),
    (2359, 1026),
    (2475, 984),
    (2545, 950),
    (2583, 931),
    (2609, 916),
    (2631, 899),
]
obj_ch4_bottom_fances_top = data["bottom_fances_top"][:15]
assert len(distorted_img_ch4_bottom_fances_top) == len(obj_ch4_bottom_fances_top)

distorted_img_ch4_bottom_fances_bar_top = [
    (139, 806),
    (169, 817),
    (219, 832),
    (276, 848),
    (391, 876),
    (591, 915),
    (1040, 970),
    (1617, 988),
    (2080, 968),
    (2346, 933),
    (2464, 910),
    (2538, 890),
    (2574, 878),
    (2604, 868),
    (2625, 860),
]
obj_ch4_bottom_fances_bar_top = data["bottom_fances_bar_top"][:15]
assert len(distorted_img_ch4_bottom_fances_bar_top) == len(
    obj_ch4_bottom_fances_bar_top
)

distorted_img_ch4_cell_bottom = [
    (931, 719),
    (1101, 620),
    (1249, 617),
    (1403, 622),
    (1557, 623),
    (1700, 630),
    (1833, 641),
    (1953, 649),
    (2054, 659),
    (2139, 667),
    (2207, 676),
    (2298, 758),
    (2381, 763),
]
obj_ch4_cell_bottom = data["bottom_cells_first_set_bottom"]
obj_ch4_cell_bottom.append(data["bottom_cells_second_set_bottom"][0])
assert len(distorted_img_ch4_cell_bottom) == len(obj_ch4_cell_bottom)

distorted_img_ch4_cell_top = [
    (900, 584),
    (1068, 577),
    (1226, 574),
    (1403, 576),
    (1578, 579),
    (1737, 593),
    (1878, 602),
    (2004, 622),
    (2107, 629),
    (2191, 643),
    (2259, 651),
    (2330, 662),
    (2412, 679),
    (2436, 684),
    (2486, 698),
    (2507, 704),
    (2524, 709),
    (2538, 715),
    (2551, 719),
    (2564, 724),
]
obj_ch4_cell_top = data["bottom_cells_first_set_top"]
obj_ch4_cell_top.extend(data["bottom_cells_second_set_top"][0:2])
obj_ch4_cell_top.extend(data["bottom_cells_second_set_top"][3:9])
assert len(distorted_img_ch4_cell_top) == len(obj_ch4_cell_top)

distorted_img_ch4_middle_cell_bottom = [
    (1122, 428),
    (1235, 385),
    (1323, 383),
    (1413, 384),
    (1502, 385),
    (1591, 390),
    (1675, 397),
    (1756, 403),
    (1903, 423),
    (1973, 433),
    (2129, 464),
    (2173, 477),
]
obj_ch4_middle_cell_bottom = []
obj_ch4_middle_cell_bottom.extend(data["middle_cells_bottom"][0:8])
obj_ch4_middle_cell_bottom.extend(data["middle_cells_bottom"][9:11])
obj_ch4_middle_cell_bottom.extend(data["middle_cells_bottom"][14:16])
assert len(distorted_img_ch4_middle_cell_bottom) == len(obj_ch4_middle_cell_bottom)

distorted_img_ch4_middle_cell_top = [
    (1110, 357),
    (1225, 346),
    (1316, 346),
    (1413, 346),
    (1512, 347),
    (1605, 354),
    (1697, 363),
    (1780, 371),
    (1932, 393),
    (1999, 406),
    (2111, 430),
    (2160, 442),
    (2202, 457),
    (2242, 470),
    (2274, 483),
    (2306, 494),
]
obj_ch4_middle_cell_top = data["middle_cells_top"][0:8]
obj_ch4_middle_cell_top.extend(data["middle_cells_top"][9:11])
obj_ch4_middle_cell_top.extend(data["middle_cells_top"][12:18])
assert len(distorted_img_ch4_middle_cell_top) == len(obj_ch4_middle_cell_top)

ch4_img = []
ch4_img.extend(distorted_img_ch4_bottom_fances_top)
ch4_img.extend(distorted_img_ch4_bottom_fances_bar_top)
ch4_img.extend(distorted_img_ch4_cell_bottom)
ch4_img.extend(distorted_img_ch4_cell_top)
ch4_img.extend(distorted_img_ch4_middle_cell_bottom)
ch4_img.extend(distorted_img_ch4_middle_cell_top)
ch4_img = np.array(ch4_img, dtype=np.float32)

ch4_obj = []
ch4_obj.extend(obj_ch4_bottom_fances_top)
ch4_obj.extend(obj_ch4_bottom_fances_bar_top)
ch4_obj.extend(obj_ch4_cell_bottom)
ch4_obj.extend(obj_ch4_cell_top)
ch4_obj.extend(obj_ch4_middle_cell_bottom)
ch4_obj.extend(obj_ch4_middle_cell_top)
ch4_obj = np.array(ch4_obj, dtype=np.float32)


distorted_img_ch8_bottom_fances_top = [
    (188, 926),
    (267, 954),
    (395, 997),
    (707, 1065),
    (1205, 1109),
    (1842, 1092),
    (2184, 1043),
    (2365, 995),
    (2455, 965),
    (2510, 941),
    (2538, 926),
]
obj_ch8_bottom_fances_top = data["top_fances_top"][-11:][::-1]
assert len(distorted_img_ch8_bottom_fances_top) == len(obj_ch8_bottom_fances_top)

distorted_img_ch8_bottom_fances_bar_top = [
    (273, 878),
    (412, 902),
    (712, 941),
    (1208, 969),
    (1833, 966),
    (2171, 939),
    (2360, 916),
    (2446, 899),
    (2499, 887),
    (2537, 877),
]
obj_ch8_bottom_fances_bar_top = data["top_fances_bar_top"][-10:][::-1]
assert len(distorted_img_ch8_bottom_fances_bar_top) == len(
    obj_ch8_bottom_fances_bar_top
)

distorted_img_ch8_cell_bottom = [
    (748, 710),
    (896, 611),
    (1035, 606),
    (1181, 601),
    (1331, 604),
    (1486, 605),
    (1635, 616),
    (1766, 625),
    (1881, 633),
    (1984, 643),
    (2102, 739),
]
obj_ch8_cell_bottom = data["top_cells_second_set_bottom"][-11:][::-1]
assert len(distorted_img_ch8_cell_bottom) == len(obj_ch8_cell_bottom)

distorted_img_ch8_cell_top = [
    (708, 592),
    (846, 576),
    (998, 569),
    (1158, 564),
    (1333, 562),
    (1506, 566),
    (1671, 575),
    (1810, 588),
    (1933, 603),
    (2038, 615),
    (2144, 631),
    (2263, 658),
    (2302, 661),
    (2343, 672),
    (2404, 690),
    (2427, 696),
    (2448, 702),
    (2480, 714),
    (2493, 719),
    (2505, 725),
    (2521, 734),
]
obj_ch8_cell_top = []
obj_ch8_cell_top.extend(data["top_cells_second_set_top"][::-1])
obj_ch8_cell_top.extend(data["top_cells_first_set_top"][:4][::-1])
obj_ch8_cell_top.extend(data["top_cells_first_set_top"][5:8][::-1])
obj_ch8_cell_top.extend(data["top_cells_first_set_top"][-3:][::-1])
assert len(distorted_img_ch8_cell_top) == len(obj_ch8_cell_top)

distorted_img_ch8_middle_cell_bottom = [
    (824, 444),
    (991, 388),
    (1075, 382),
    (1161, 377),
    (1251, 376),
    (1343, 375),
    (1430, 378),
    (1522, 383),
    (1607, 388),
    (1687, 395),
    (1763, 405),
    (1901, 426),
    (2106, 473),
    (2148, 485),
    (2214, 507),
    (2271, 528),
    (2295, 538),
    (2318, 546),
]
obj_ch8_middle_cell_bottom = []
obj_ch8_middle_cell_bottom.append(data["middle_cells_bottom1"][-1])
obj_ch8_middle_cell_bottom.extend(data["middle_cells_bottom1"][-12:-2][::-1])
obj_ch8_middle_cell_bottom.append(data["middle_cells_bottom1"][-14])
obj_ch8_middle_cell_bottom.extend(data["middle_cells_bottom1"][-19:-17][::-1])
obj_ch8_middle_cell_bottom.append(data["middle_cells_bottom1"][-21])
obj_ch8_middle_cell_bottom.extend(data["middle_cells_bottom1"][1:4][::-1])
assert len(distorted_img_ch8_middle_cell_bottom) == len(obj_ch8_middle_cell_bottom)

distorted_img_ch8_middle_cell_top = [
    (803, 372),
    (965, 353),
    (1054, 348),
    (1148, 340),
    (1245, 338),
    (1343, 340),
    (1437, 345),
    (1533, 347),
    (1625, 360),
    (1713, 363),
    (1791, 373),
    (1865, 386),
    (1933, 398),
    (2046, 424),
    (2095, 437),
    (2139, 452),
    (2176, 465),
    (2212, 478),
    (2244, 490),
    (2271, 502),
    (2296, 512),
    (2320, 523),
    (2342, 533),
]
obj_ch8_middle_cell_top = []
obj_ch8_middle_cell_top.append(data["middle_cells_top1"][-1])
obj_ch8_middle_cell_top.extend(data["middle_cells_top1"][-14:-2][::-1])
obj_ch8_middle_cell_top.extend(data["middle_cells_top1"][-25:-15][::-1])
assert len(distorted_img_ch8_middle_cell_top) == len(obj_ch8_middle_cell_top)

ch8_img = []
ch8_img.extend(distorted_img_ch8_bottom_fances_top)
ch8_img.extend(distorted_img_ch8_bottom_fances_bar_top)
ch8_img.extend(distorted_img_ch8_cell_bottom)
ch8_img.extend(distorted_img_ch8_cell_top)
ch8_img.extend(distorted_img_ch8_middle_cell_bottom)
ch8_img.extend(distorted_img_ch8_middle_cell_top)
ch8_img = np.array(ch8_img, dtype=np.float32)

ch8_obj = []
ch8_obj.extend(obj_ch8_bottom_fances_top)
ch8_obj.extend(obj_ch8_bottom_fances_bar_top)
ch8_obj.extend(obj_ch8_cell_bottom)
ch8_obj.extend(obj_ch8_cell_top)
ch8_obj.extend(obj_ch8_middle_cell_bottom)
ch8_obj.extend(obj_ch8_middle_cell_top)
ch8_obj = np.array(ch8_obj, dtype=np.float32)


distorted_img_ch6_bottom_fances_top = [
    (109, 857),
    (138, 875),
    (178, 895),
    (243, 927),
    (352, 967),
    (601, 1031),
    (1048, 1093),
    (1718, 1090),
    (2119, 1043),
    (2327, 993),
    (2446, 952),
    (2509, 927),
]
obj_ch6_bottom_fances_top = data["top_fances_top"][2:14][::-1]
assert len(distorted_img_ch6_bottom_fances_top) == len(obj_ch6_bottom_fances_top)

distorted_img_ch6_bottom_fances_bar_top = [
    (110, 811),
    (136, 821),
    (182, 832),
    (250, 850),
    (367, 873),
    (618, 912),
    (1061, 947),
    (1718, 952),
    (2109, 929),
    (2319, 902),
    (2440, 882),
    (2501, 869),
    (2545, 857),
]
obj_ch6_bottom_fances_bar_top = data["top_fances_bar_top"][1:14][::-1]
assert len(distorted_img_ch6_bottom_fances_bar_top) == len(
    obj_ch6_bottom_fances_bar_top
)

distorted_img_ch6_cell_bottom = [
    (434, 707),
    (685, 607),
    (787, 604),
    (1044, 595),
    (1194, 593),
    (1348, 598),
    (1499, 596),
    (1647, 605),
    (1806, 708),
]
obj_ch6_cell_bottom = []
obj_ch6_cell_bottom.append(data["top_cells_first_set_bottom"][-1])
obj_ch6_cell_bottom.extend(data["top_cells_first_set_bottom"][6:8][::-1])
obj_ch6_cell_bottom.extend(data["top_cells_first_set_bottom"][:6][::-1])
assert len(distorted_img_ch6_cell_bottom) == len(obj_ch6_cell_bottom)

distorted_img_ch6_cell_top = [
    (215, 651),
    (236, 645),
    (259, 639),
    (288, 633),
    (313, 627),
    (397, 612),
    (468, 595),
    (539, 582),
    (628, 575),
    (735, 562),
    (861, 552),
    (1011, 548),
    (1173, 545),
    (1347, 547),
    (1520, 549),
    (1683, 564),
    (1846, 579),
]
obj_ch6_cell_top = []
obj_ch6_cell_top.extend(data["top_cells_second_set_top"][:5][::-1])
obj_ch6_cell_top.extend(data["top_cells_first_set_top"][::-1])
assert len(distorted_img_ch6_cell_top) == len(obj_ch6_cell_top)

distorted_img_ch6_middle_cell_bottom = [
    (847, 386),
    (996, 371),
    (1082, 358),
    (1166, 358),
    (1255, 362),
    (1348, 362),
    (1443, 365),
    (1532, 369),
    (1631, 417),
]
obj_ch6_middle_cell_bottom = []
obj_ch6_middle_cell_bottom.append(data["middle_cells_bottom1"][9])
obj_ch6_middle_cell_bottom.extend(data["middle_cells_bottom1"][:8][::-1])
assert len(distorted_img_ch6_middle_cell_bottom) == len(obj_ch6_middle_cell_bottom)

distorted_img_ch6_middle_cell_top = [
    (422, 442),
    (457, 433),
    (495, 422),
    (533, 410),
    (579, 400),
    (745, 365),
    (971, 337),
    (1062, 332),
    (1152, 326),
    (1249, 325),
    (1349, 329),
    (1451, 331),
    (1546, 335),
    (1645, 342),
]
obj_ch6_middle_cell_top = []
obj_ch6_middle_cell_top.extend(data["middle_cells_top1"][13:18][::-1])
obj_ch6_middle_cell_top.append(data["middle_cells_top1"][10])
obj_ch6_middle_cell_top.extend(data["middle_cells_top1"][:8][::-1])
assert len(distorted_img_ch6_middle_cell_top) == len(obj_ch6_middle_cell_top)

ch6_img = []
ch6_img.extend(distorted_img_ch6_bottom_fances_top)
ch6_img.extend(distorted_img_ch6_bottom_fances_bar_top)
ch6_img.extend(distorted_img_ch6_cell_bottom)
ch6_img.extend(distorted_img_ch6_cell_top)
ch6_img.extend(distorted_img_ch6_middle_cell_bottom)
ch6_img.extend(distorted_img_ch6_middle_cell_top)
ch6_img = np.array(ch6_img, dtype=np.float32)

ch6_obj = []
ch6_obj.extend(obj_ch6_bottom_fances_top)
ch6_obj.extend(obj_ch6_bottom_fances_bar_top)
ch6_obj.extend(obj_ch6_cell_bottom)
ch6_obj.extend(obj_ch6_cell_top)
ch6_obj.extend(obj_ch6_middle_cell_bottom)
ch6_obj.extend(obj_ch6_middle_cell_top)
ch6_obj = np.array(ch6_obj, dtype=np.float32)