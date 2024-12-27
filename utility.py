import datetime
from datetime import datetime

#datetime >> stringformat
def cv_format(d):
    fstr = "%Y년 %m월 %d일 %H시 %M분 %S초"
    return d.strftime(fstr)

#millsecond >> datetime
def cv_mill2date(millsec):
    return datetime.fromtimestamp(millsec)

#datetime >> millsecond
def cv_date2milli(d):
    return d.timestamp()

#stringformat >> datetime
def cv_str2date(dstr):
    fstr = "%Y년 %m월 %d일 %H시 %M분 %S초"
    return datetime.strptime(dstr,fstr)
    
#날짜시간데이터로 변경
def GetCompare(start_mill, end_mill):
    # 밀리초 단위 차이를 초로 변환하여 gap 계산
    gap = (end_mill - start_mill) /1000 # 밀리초를 초로 변환
    gap_day = 60 * 60 * 24
    gap_hour = 60 * 60
    gap_min = 60
    gapday = gap // gap_day  # 일 계산
    gaphour = (gap - gapday * gap_day) // gap_hour  # 시 계산
    gapmin = (gap - gapday * gap_day - gaphour * gap_hour) // gap_min  # 분 계산
    gapsec = gap - gapday * gap_day - gaphour * gap_hour - gapmin * gap_min  # 초 계산
    return {"gapday": gapday, "gaphour": gaphour, "gapmin": gapmin, "gapsec": gapsec}

tdata = 1388070000000/1000
d = cv_mill2date(tdata)
dstr = cv_format(d)
d = cv_str2date(dstr)
dmil = cv_date2milli(d)
current_time = datetime.now().timestamp()
# print(GetCompare(dmil, current_time))

if __name__ == "__main__":
    pass
