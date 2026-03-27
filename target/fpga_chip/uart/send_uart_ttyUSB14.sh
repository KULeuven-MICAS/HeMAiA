# /bin/bash

if [ "$#" -ne 0 ] && [ "$#" -ne 1 ];then
	echo "Usage: send_uart [filename]"
	exit 1
else

	if [ "$#" -ne 1 ];then
		FILE=app.bin
	else
		FILE=$1
	fi

	stty -F /dev/ttyUSB14 cs8 500000 ignbrk -brkint -imaxbel -opost -onlcr -isig -icanon -iexten -echo -echoe -echok -echoctl -echoke noflsh -ixon crtscts

	echo -n 3 > /dev/ttyUSB14

	sx -k "$FILE" < /dev/ttyUSB14 > /dev/ttyUSB14

fi

