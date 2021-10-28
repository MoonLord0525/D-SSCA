import csv


def ShapeRToCsv(path, shapes, seq_len):
    """
        将DNAShapeR工具得到的shape样本，保存至csv格式的文件

        :parameters
            shapes是列表，其中存储待处理的shape类型的名称，如Roll等
            path的格式，需要文件名(如Test_shape.data)，但不需要.shape_name

        结果将保存在相同的文件夹中
        一次只能处理一个DataSet
    """
    for shape in shapes:
        i_file = open(path + '.' + shape)
        o_file = csv.writer(open(path + shape + '.csv', 'w', newline=''))

        """
            write header
        """
        row = []
        for i in range(seq_len):
            row.append(i+1)

        for line in i_file.readlines():
            """
                文件格式:
                    >1
                    NA,NA,4.96,.......,4.92,NA,NA
            """
            line = line.replace('\n', '')
            if line[0] == '>':
                o_file.writerow(row)
                row = []
            else:
                line = line.split(',')
                for char in line:
                    if char == 'NA':
                        row.append(float(0))
                    else:
                        row.append(float(char))
    print('\033[32m' + 'success!')


ShapeRToCsv(path='wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk\\train_shape.data',
            shapes=['EP', 'HelT', 'MGW', 'ProT', 'Roll'], seq_len=101)