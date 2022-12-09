video_name="test3"
f = open(f"{video_name}-output.log", "r")


lines=f.readlines()
get_eight_point_2=[]
get_eight_point_1=[]
object_motion_vector_list=[]
for i,line in enumerate(lines):
    line=line.rstrip()
    line+=","
    line+='\n'
    if i%4==3:
        get_eight_point_2.append(line)
    elif i%4==2:
        get_eight_point_1.append(line)
    elif i%4==1:
        object_motion_vector_list.append(line)

file1 = open(f'{video_name}-get_eight_point_2.txt', 'w')
file1.writelines(get_eight_point_2)
file1.close()
file2 = open(f'{video_name}-get_eight_point_1.txt', 'w')
file2.writelines(get_eight_point_1)
file2.close()
file3 = open(f'{video_name}-object_motion_vector_list.txt', 'w')
file3.writelines(object_motion_vector_list)
file3.close()