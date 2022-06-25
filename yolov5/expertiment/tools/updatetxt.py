
filename ="E:\kg\data\SONAR_VOC_MulitScale\VOC2007\ImageSets\Main/"+ "val.txt"

with open(filename, "r") as file:
    lines = file.readlines()

with open(filename, "w", encoding="utf-8") as f_w:
    for line in lines:
        line_ = line.strip('\n')
        print(line)
        item = line_ + "_32_32"
        item = item+'\n'
        f_w.write(item)

        item = line_ + "_32_32_Angle_90"
        item = item+'\n'
        f_w.write(item)

        item = line_ + "_32_32_Angle_180"
        item = item+'\n'
        f_w.write(item)

        item = line_ + "_32_32_Angle_270"
        item = item+'\n'
        f_w.write(item)

        item = line_ + "_32_64"
        item = item+'\n'
        f_w.write(item)

        item = line_ + "_64_32"
        item = item+'\n'
        f_w.write(item)

        item = line_ + "_64_64"
        item = item+'\n'
        f_w.write(item)

        item = line_ + "_64_64_Angle_90"
        item = item+'\n'
        f_w.write(item)

        item = line_ + "_64_64_Angle_180"
        item = item+'\n'
        f_w.write(item)

        item = line_ + "_64_64_Angle_270"
        item = item+'\n'
        f_w.write(item)

