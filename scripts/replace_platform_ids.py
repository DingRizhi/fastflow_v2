import glob
import os

a = '''
1	1
2	2
3	3
4	4
5	5
6	6
7	7
8	8
9	9
10	10
11	11
12	12
13	13
14	14
-	15
15	16
-	17
16	18
17	19
-	20
18	21
-	22
19	23
20	24
21	25
22	26
23	27
24	28
25	29
26	30
27	31
28	32
29	33
30	34
31	35
32	36
33	37
34	38
35	39
36	40
37	41
38	42
39	43
40	44
41	45
42	46
43	47
44	48
45	49
46	50
47	51
48	52
49	53
50	54
51	55
52	56
53	57
-	58
54	59
-	60
55	61
56	62
57	63
58	64
-	65
59	66
60	67
61	68
62	69
63	70
64	71
65	72
66	73
67	74
68	75
69	76
70	77
71	78
72	79
73	80
74	81
75	82
76	83
77	84
78	85
79	86
80	87
81	88
82	89
83	90
84	91
85	92
86	93
87	94
88	95
89	96
90	97
91	98
92	99
93	100
-	101
94	102
95	103
-	104
96	105
97	106
-	107
98	108
-	109
99	110
100	111
101	112
102	113
103	114
104	115
105	116
106	117
107	118
108	119
109	120
110	121
111	122
112	123
113	124
114	125
115	126
116	127
117	128
118	129
119	130
120	131
121	132
122	133
123	134
124	135
125	136
126	137
127	138
128	139
129	140
130	141
131	142
132	143
133	144
134	145
135	-
136	146
137	147
138	148
139	149
140	150
141	151
142	152
143	153
144	154
145	155
146	156
147	157
148	158
149	159
150	160
151	161
152	162
153	163
154	164
155	165
156	166
157	167
158	168
159	169
160	170
161	171
162	172
163	173
164	174
165	-
166	-
167	175
168	176
169	177
170	178
171	-
172	179
173	180
174	181
175	182
176	183
177	184
178	185
179	186
180	187
181	188
182	189
183	190
184	191
185	192
186	193
187	194
188	195
189	196
190	197
191	198
192	199
193	200
-	201
194	202
195	-
196	-
197	203
198	-
199	204
200	205
201	206
202	207
203	-
204	208
205	209
206	210
207	211
208	212
209	213
210	214
211	215
'''

convert_dict = {}
old_miss_matched = []
new_miss_matched = []
b = a.split("\n")
for i in b:
    if i == "":
        continue

    c = i.split("\t")
    if c[0] == "-":
        old_miss_matched.append([c[0], c[1]])
        continue
    if c[1] == "-":
        new_miss_matched.append([c[0], c[1]])
        continue
    convert_dict[c[0]] = c[1]

old_miss_matched = sorted(old_miss_matched)
new_miss_matched = sorted(new_miss_matched)

print(f"old miss match: {old_miss_matched}")
print(f"new miss match: {new_miss_matched}")


def get_new_ids(old_ids):
    new_ids = []
    for i in old_ids:
        if str(i) not in convert_dict:
            print(f"{i} not in convert_dict")
            continue
        new_i = convert_dict[str(i)]
        new_ids.append(int(new_i))
        new_ids = sorted(new_ids)
    print(new_ids)


def rename_val_image_platform_id(img_dir):
    image_path_list = glob.glob(f"{img_dir}/*.jpg")
    print(f"total images: {len(image_path_list)}")

    for image_path in image_path_list:
        dir_path = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)

        img_pure_name = os.path.splitext(base_name)[0]
        ss = img_pure_name.split("-")
        img_platform_id = ss[-1]
        if str(img_platform_id) not in convert_dict:
            print(f"{image_path}'s platform id--{img_platform_id} not in convert_dict, remove")
            os.remove(image_path)
        new_img_platform_id = convert_dict[str(img_platform_id)]
        ss[-1] = new_img_platform_id

        new_image_name = "-".join(ss) + ".jpg"

        new_img_path = os.path.join(dir_path, new_image_name)

        os.rename(image_path, new_img_path)


if __name__ == '__main__':
    # rename_val_image_platform_id("/home/log/PycharmProjects/triton_deploy_cloud/triton_template/test_data/val_02_img")
    # get_new_ids([20, 23, 24, 26, 27, 30, 31, 33, 34, 36, 39, 60, 63, 64, 66, 67, 70, 71, 73, 74, 76, 79, 103, 104, 106, 107, 110, 111, 113, 114, 116])
    pass
