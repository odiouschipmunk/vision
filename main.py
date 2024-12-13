def main():
    import csv
    import ast
    frames=[]
    with open('output/final.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            frames.append(row)
    #goes through frame --> player1 --> first keypoint --> x
    print(ast.literal_eval(frames[1][1])[0][0])
if __name__ == "__main__":
    main()
