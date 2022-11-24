# human-interface-media-2022-2
Human Interface Media Project 2022-2
20184757 주영석

개발 환경 및 설치된 패키지 버전
- MacBook Air, M2, 2022
- macOS Ventura 13.0.1
- python 3.7
	- Pillow 9.2.0
	- numpy 1.23.2
	- matplotlib 3.5.3
	- tqdm 4.64.1

프로그램 실행 방법
1. 현재 경로에서 터미널을 실행한다.
2. "python3 ncc.py [파일명1] [파일명2]" 를 입력한다. 파일명에는 원하는 이미지의 경로를 입력한다. 템플릿 이미지의 경우 가로와 세로 픽셀의 길이가 동일한 정사각형 이미지여야 한다.
3. 찾으려는 이미지의 회전 각도의 목록을 스페이스로 구분하여 입력한 후, 엔터를 누른다. 아무 입력도 하지 않고 엔터를 누를 경우 회전이 적용되지 않는다. (입력 예: "0 45 270")
4. 찾으려는 이미지의 크기 조정 계수 목록을 스페이스로 구분하여 입력한 후, 엔터를 누른다. 아무 입력도 하지 않고 엔터를 누를 경우 크기 조정이 적용되지 않는다. (입력 예: "1.0 0.5 1.2")
5. 프로그램 실행이 완료될 때까지 기다린다.
6. 새 창에 출력되는 결과 이미지를 확인한다. 회전 및 크기 조정이 적용된 템플릿 이미지의 목록, NCC 결과 이미지, 최종 결과 이미지가 각각 출력된다. 최종 결과 이미지는 ./result 경로에 별도의 파일로 저장된다.

예시 파일
1. ./example/pic1.jpg (참조 이미지)
	1-1. ./example/pic1-1.jpg => result
	1-2. ./example/pic1-2.jpg
	1-3. ./example/pic1-3.jpg
2. ./example/pic2.jpg (참조 이미지)
	2-1. ./example/pic2-1.jpg
	2-2. ./example/pic2-2.jpg => result
	2-3. ./example/pic2-3.jpg
3. ./example/pic3.jpg (참조 이미지)
	3-1. ./example/pic3-1.jpg => result