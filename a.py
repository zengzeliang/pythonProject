from queue import PriorityQueue as PQ


def countSubstrings(s: str) -> int:
    ans = 0

    contact = ""

    for ch in s:
        contact += ("#" + ch)
    contact += "#"

    for i in range(len(contact)):
        x = i
        y = i
        while x >= 0 and y < len(contact) and contact[x] == contact[y]:
            x -= 1
            y += 1
        if (y - x) // 2 > 1:

            ans += (y - x) // 2

    return ans // 2

def loadword2Id():
    word2idx = {}
    f = open("./model/vocab.txt", encoding='utf-8')
    while True:
        line = f.readline()
        if line.strip(""):
            splits = line.split(":")
            if len(splits) == 3:
                word2idx[line[:line.index(splits[2]) - 1]] = int(splits[2])
                print(line[:line.index(splits[2]) - 1] , " ", splits[2])
            else:
                word2idx[splits[0]] = int(splits[1])
        else:
            break
    f.close()
    return word2idx


if __name__ == '__main__':

    # ans = countSubstrings("abc")
    # print(ans)
    word2idx = loadword2Id()
    # print(word2idx)


