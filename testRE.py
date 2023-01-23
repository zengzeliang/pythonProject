import re

from nebula.createKG import parseSightLevel

if __name__ == '__main__':
    sightLevel = "你知道吗aaaaAAAA"

    ans = parseSightLevel("AAAA级[3]")

    print(ans)
