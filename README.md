<div style="text-align:center">

# GPGPU Assignment

## A simple Edge Detector Using:

<img src="https://img.shields.io/badge/-C++-333333?style=flat-square&logo=c%2B%2B" alt="cxx" height="30"/>
<img src="https://img.shields.io/badge/-OpenGL-333333?style=flat-square&logo=opengl" alt="opengl"  height="30"/>
<img src="https://img.shields.io/badge/-Cuda-333333?style=flat-square&logo=nvidia" alt="cuda"  height="30"/>
<img src="https://img.shields.io/badge/-opencl-333333?style=flat-square&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4NCjwhLS0gR2VuZXJhdG9yOiBBZG9iZSBJbGx1c3RyYXRvciAyMi4wLjEsIFNWRyBFeHBvcnQgUGx1Zy1JbiAuIFNWRyBWZXJzaW9uOiA2LjAwIEJ1aWxkIDApICAtLT4NCjxzdmcgdmVyc2lvbj0iMS4xIiBpZD0iVnVsa2FuIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgd2lkdGg9IjExMDBweCIgaGVpZ2h0PSI1MDBweCIgdmlld0JveD0iMCAwIDExMDAgNTAwIiBzdHlsZT0iZW5hYmxlLWJhY2tncm91bmQ6bmV3IDAgMCAxMTAwIDUwMDsiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4NCgkuc3Qwe2ZpbGw6I0FCRDAzODt9DQoJLnN0MXtmaWxsOiNFRTMzMkQ7fQ0KCS5zdDJ7ZmlsbDojRjJCQTFBO30NCgkuc3Qze2ZpbGw6IzNEQUUyQjt9DQoJLnN0NHtmaWxsOiMwMTAxMDE7fQ0KPC9zdHlsZT4NCjxwYXRoIGNsYXNzPSJzdDAiIGQ9Ik00NzcuOSw5OC4zYzE0LjYtMjIuOCwzMS44LTQ5LjgsMzcuNC01OC42Yy02My45LDQuOC0xMjMsMjEuNC0xNzMsNDYuN2M3LjcsMTQuNCwxNy43LDMyLjUsMjUuNCw0Ni41DQoJYzI2LjUtOS45LDU1LjMtMTYuNyw4NS43LTE5LjZDNDYzLjQsMTEyLjMsNDcyLjQsMTA2LjgsNDc3LjksOTguM3oiLz4NCjxwYXRoIGNsYXNzPSJzdDEiIGQ9Ik03MTEuMiwxOTAuNWMxMy45LDYuNiwyNy40LDguNiw0Ni4zLDguNmMzNC42LDAsOTkuOSwwLjEsMTM1LjgsMC4yYzAsMCwwLDAsMCwwDQoJYy0zMS42LTUyLjUtODUuOC05Ni4yLTE1My43LTEyNC44Yy0yNC4xLDE2LjEtOTMuMSw2Mi4zLTEwNC4xLDY5LjZDNjc5LjEsMTY1LjQsNjk3LjUsMTg0LDcxMS4yLDE5MC41eiIvPg0KPHBhdGggY2xhc3M9InN0MiIgZD0iTTUyOC41LDM4LjljLTEuOSw1LjYtMTguOSw1Ni42LTI0LjMsNzMuMWMyNy40LDEuMiw1My44LDUuNCw3OC41LDEyLjNjMTQuNyw0LjEsMzAuNSwyLjYsNDQuMS00LjRsOTkuNi01MC44DQoJQzY3NC45LDQ5LjQsNjE2LjMsMzguMiw1NTQsMzguMkM1NDUuNCwzOC4yLDUzNi45LDM4LjUsNTI4LjUsMzguOXoiLz4NCjxwYXRoIGNsYXNzPSJzdDMiIGQ9Ik0yMDUuNiwyMTIuOWwzMi4zLDAuMWM4LjcsMCwxNy0zLjYsMjIuOS0xMGMxNy41LTE5LjEsMzktMzUuOSw2My43LTQ5LjhjOC44LTQuOSwxMy43LTE0LjgsMTIuMS0yNC43DQoJYy0yLjQtMTUuNi01LTMxLjctNS43LTM2QzI3NC42LDEyMywyMzAuOCwxNjQuNiwyMDUuNiwyMTIuOXoiLz4NCjxnPg0KCTxwb2x5Z29uIGNsYXNzPSJzdDQiIHBvaW50cz0iOTg3LjksMjM5LjUgOTgwLjUsMjM5LjUgOTgwLjUsMjU4LjggOTc0LDI1OC44IDk3NCwyMzkuNSA5NjYuNSwyMzkuNSA5NjYuNSwyMzMuOSA5ODcuOSwyMzMuOSANCgkJOTg3LjksMjM5LjUgCSIvPg0KCTxwb2x5bGluZSBjbGFzcz0ic3Q0IiBwb2ludHM9IjEwMTguNiwyMzMuOSAxMDE4LjYsMjU4LjggMTAxMi41LDI1OC44IDEwMTIuNSwyMzkuOSAxMDEyLjQsMjM5LjkgMTAwNy4yLDI1OC44IDEwMDIuMiwyNTguOCANCgkJOTk3LDIzOS45IDk5Ni45LDIzOS45IDk5Ni45LDI1OC44IDk5MC44LDI1OC44IDk5MC44LDIzMy45IDk5MC44LDIzMy45IDEwMDAuNCwyMzMuOSAxMDA0LjcsMjUwLjUgMTAwNC43LDI1MC41IDEwMDksMjMzLjkgCSIvPg0KPC9nPg0KPGc+DQoJPHBhdGggY2xhc3M9InN0NCIgZD0iTTIxNi43LDI2MC4zYy03LjktOS4yLTE3LjYtMTYuNS0yOS4xLTIxLjhjLTExLjQtNS40LTI0LjQtOC4xLTM4LjgtOC4xYy0xNC40LDAtMjcuMywyLjctMzguOCw4LjENCgkJYy0xMS40LDUuNC0yMS4xLDEyLjctMjkuMSwyMS44Yy03LjksOS4yLTE0LDE5LjktMTguMiwzMi4xYy00LjIsMTIuMi02LjQsMjUuMi02LjQsMzguOGMwLDEzLjgsMi4xLDI2LjksNi40LDM5LjENCgkJYzQuMiwxMi4yLDEwLjMsMjIuOSwxOC4yLDMyLjFjNy45LDkuMiwxNy42LDE2LjQsMjkuMSwyMS43YzExLjQsNS4zLDI0LjQsOCwzOC44LDhjMTQuNCwwLDI3LjMtMi43LDM4LjgtOA0KCQljMTEuNC01LjMsMjEuMS0xMi41LDI5LjEtMjEuN2M3LjktOS4yLDE0LTE5LjksMTguMi0zMi4xYzQuMi0xMi4yLDYuNC0yNS4zLDYuNC0zOS4xYzAtMTMuNy0yLjEtMjYuNi02LjQtMzguOA0KCQlDMjMwLjcsMjgwLjIsMjI0LjYsMjY5LjUsMjE2LjcsMjYwLjN6IE0yMDQuMSwzNTguMmMtMi4zLDguOC01LjcsMTYuNy0xMC40LDIzLjZjLTQuNyw2LjktMTAuOCwxMi41LTE4LjIsMTYuNg0KCQljLTcuNSw0LjEtMTYuNCw2LjItMjYuNiw2LjJjLTEwLjMsMC0xOS4xLTIuMS0yNi42LTYuMmMtNy41LTQuMS0xMy42LTkuNy0xOC4yLTE2LjZjLTQuNy02LjktOC4yLTE0LjgtMTAuNC0yMy42DQoJCWMtMi4zLTguOC0zLjQtMTcuOC0zLjQtMjdzMS4xLTE4LjIsMy40LTI3YzIuMy04LjgsNS43LTE2LjcsMTAuNC0yMy42YzQuNy02LjksMTAuOC0xMi40LDE4LjItMTYuNmM3LjUtNC4xLDE2LjQtNi4yLDI2LjYtNi4yDQoJCWMxMC4zLDAsMTkuMSwyLjEsMjYuNiw2LjJjNy41LDQuMSwxMy42LDkuNywxOC4yLDE2LjZjNC43LDYuOSw4LjIsMTQuOCwxMC40LDIzLjZjMi4zLDguOCwzLjQsMTcuOCwzLjQsMjcNCgkJUzIwNi4zLDM0OS40LDIwNC4xLDM1OC4yeiIvPg0KCTxwYXRoIGNsYXNzPSJzdDQiIGQ9Ik0zODQuNSwzMDZjLTUuNC02LjctMTIuMi0xMi0yMC40LTE1LjljLTguMi0zLjktMTcuOC01LjgtMjguOC01LjhjLTksMC0xNy40LDEuOC0yNS4xLDUuMw0KCQljLTcuOCwzLjUtMTMuOCw5LjMtMTguMSwxNy40aC0wLjVWMjg4aC0yOS4ydjE5MWgzMC44di02OS42aDAuNWMyLjMsMy44LDUuMyw3LjEsOC44LDkuOGMzLjUsMi44LDcuMyw1LjEsMTEuNSw2LjkNCgkJYzQuMSwxLjgsOC41LDMuMSwxMywzLjljNC41LDAuOCw5LDEuMiwxMy41LDEuMmMxMC4zLDAsMTkuMi0yLDI2LjgtNi4xYzcuNi00LDEzLjgtOS40LDE4LjgtMTYuMmM1LTYuNyw4LjctMTQuNSwxMS4xLTIzLjMNCgkJYzIuNC04LjgsMy42LTE3LjksMy42LTI3LjNjMC0xMC4zLTEuNC0xOS45LTQuMS0yOC45QzM5NCwzMjAuNiwzODkuOSwzMTIuNywzODQuNSwzMDZ6IE0zNjcuOSwzNzUuOGMtMS40LDUuOC0zLjYsMTEuMS02LjgsMTUuNw0KCQljLTMuMiw0LjYtNy4yLDguMy0xMiwxMS4yYy00LjksMi45LTEwLjgsNC4zLTE3LjgsNC4zYy02LjMsMC0xMS45LTEuMy0xNi42LTMuOGMtNC44LTIuNS04LjktNS45LTEyLjMtMTAuMw0KCQljLTMuNC00LjMtNi05LjUtNy43LTE1LjVjLTEuNy02LTIuNi0xMi41LTIuNi0xOS42YzAtMTQuOSwzLjMtMjYuOSw5LjktMzUuOWM2LjYtOSwxNi40LTEzLjUsMjkuMy0xMy41YzYuMywwLDExLjksMS40LDE2LjgsNC4yDQoJCWM0LjksMi44LDguOSw2LjUsMTIuMiwxMS4xczUuNyw5LjgsNy4zLDE1LjhjMS42LDUuOSwyLjQsMTIuMSwyLjQsMTguM0MzNjkuOSwzNjQsMzY5LjMsMzcwLDM2Ny45LDM3NS44eiIvPg0KCTxwYXRoIGNsYXNzPSJzdDQiIGQ9Ik01MzEuOCwzMDkuM2MtNS44LTcuNi0xMy0xMy42LTIxLjgtMTguMmMtOC43LTQuNi0xOC42LTYuOS0yOS42LTYuOWMtMTAuNSwwLTE5LjgsMi0yOC4xLDUuOQ0KCQljLTguMyw0LTE1LjQsOS4zLTIxLjIsMTYuMWMtNS45LDYuNy0xMC40LDE0LjYtMTMuNSwyMy41Yy0zLjIsOC45LTQuNywxOC4zLTQuNywyOC4yYzAsMTAuOCwxLjUsMjAuNyw0LjUsMjkuNw0KCQljMyw5LDcuNCwxNi43LDEzLjIsMjMuMmM1LjksNi41LDEzLDExLjUsMjEuNSwxNS4xYzguNSwzLjYsMTguMiw1LjQsMjkuMiw1LjRjNy42LDAsMTQuOC0xLDIxLjgtMy4xYzYuOS0yLjEsMTMuMi01LjEsMTguOC05LjINCgkJYzUuNi00LDEwLjQtOSwxNC4zLTE1YzQtNS45LDYuOC0xMi43LDguNC0yMC4yaC0yOS4yYy0yLjcsNy43LTYuOCwxMy41LTEyLjMsMTcuNGMtNS41LDMuOS0xMi44LDUuOC0yMS44LDUuOA0KCQljLTYuNSwwLTEyLjEtMS4xLTE2LjgtMy40Yy00LjctMi4yLTguNi01LjMtMTEuNi05Yy0zLjEtMy44LTUuNC04LjItNi45LTEzLjJjLTEuNS01LTIuMy0xMC4zLTIuMy0xNS43aDEwMi43DQoJCWMxLjMtMTAuNCwwLjYtMjAuNS0yLTMwLjRDNTQxLjcsMzI1LjYsNTM3LjUsMzE2LjksNTMxLjgsMzA5LjN6IE00NDMuNiwzNDUuNWMwLjItNS4yLDEuMi0xMC4xLDMuMS0xNC42YzEuOS00LjUsNC40LTguNCw3LjYtMTEuNw0KCQljMy4yLTMuMyw2LjktNS45LDExLjQtNy44YzQuNC0xLjksOS4zLTIuOCwxNC43LTIuOGM1LjIsMCwxMCwxLDE0LjIsMy4xYzQuMiwyLjEsNy44LDQuOCwxMC44LDguMWMzLDMuMyw1LjMsNy4yLDcsMTEuNw0KCQljMS43LDQuNSwyLjcsOS4yLDMuMSwxNEg0NDMuNnoiLz4NCgk8cGF0aCBjbGFzcz0ic3Q0IiBkPSJNNjczLjMsMjk2LjFjLTguOC03LjktMjAuOS0xMS45LTM2LjItMTEuOWMtOS4yLDAtMTcuNiwyLjItMjUuMSw2LjZjLTcuNiw0LjQtMTMuNywxMC41LTE4LjQsMTguMmwtMC41LTAuNQ0KCQlWMjg4aC0yOS4ydjEzOS41aDMwLjh2LTgyLjNjMC01LDAuOC05LjgsMi40LTE0LjNjMS42LTQuNSwzLjktOC40LDYuOC0xMS43YzIuOS0zLjMsNi40LTUuOSwxMC41LTcuOGM0LjEtMS45LDguOC0yLjgsMTQuMS0yLjgNCgkJYzkuMiwwLDE2LDIuNSwyMC4zLDcuNGM0LjMsNC45LDYuNywxMi45LDcsMjMuOXY4Ny43aDMwLjh2LTk1LjhDNjg2LjYsMzE1LjksNjgyLjEsMzA0LDY3My4zLDI5Ni4xeiIvPg0KCTxwYXRoIGNsYXNzPSJzdDQiIGQ9Ik04NDAuNywzNTIuMWMtMS41LDEyLTUuOCwyMS45LTEyLjYsMjkuNGMtNi43LDcuMy0xNi4yLDExLjEtMjguMiwxMS4xYy05LDAtMTYuNi0xLjctMjIuOC01LjINCgkJYy02LjItMy40LTExLjMtOC0xNS4yLTEzLjdjLTMuOS01LjctNi44LTEyLjMtOC42LTE5LjZjLTEuOC03LjMtMi43LTE0LjktMi43LTIyLjZjMC04LjEsMC45LTE2LDIuNy0yMy42DQoJCWMxLjgtNy41LDQuNy0xNC4yLDguNi0yMC4xYzMuOS01LjgsOC45LTEwLjQsMTUuMi0xMy45YzYuMi0zLjQsMTMuOS01LjIsMjIuOC01LjJjNC44LDAsOS42LDAuOCwxNCwyLjRjNC40LDEuNiw4LjUsMy44LDEyLDYuNw0KCQljMy41LDIuOCw2LjUsNi4yLDguOCw5LjljMi4zLDMuNywzLjgsNy44LDQuNSwxMi4zbDAuMywxLjloNDQuOGwtMC4zLTIuNWMtMS4zLTExLjItNC40LTIxLjMtOS4zLTMwYy00LjktOC43LTExLjItMTYuMS0xOC44LTIyDQoJCWMtNy41LTUuOS0xNi4xLTEwLjQtMjUuNi0xMy41Yy05LjUtMy4xLTE5LjctNC42LTMwLjMtNC42Yy0xNC45LDAtMjguNSwyLjYtNDAuMiw3LjljLTExLjcsNS4yLTIxLjgsMTIuNi0zMCwyMS44DQoJCWMtOC4xLDkuMi0xNC40LDIwLjEtMTguOCwzMi41Yy00LjMsMTIuNC02LjUsMjUuOC02LjUsMzkuOWMwLDEzLjgsMi4yLDI3LDYuNSwzOS4yYzQuMywxMi4yLDEwLjYsMjMsMTguOCwzMg0KCQljOC4xLDksMTguMiwxNi4zLDMwLDIxLjVjMTEuNyw1LjIsMjUuMyw3LjksNDAuMiw3LjljMTEuOSwwLDIzLTEuOSwzMi45LTUuNmMxMC0zLjcsMTguOC05LjEsMjYuMi0xNmM3LjQtNi45LDEzLjUtMTUuMywxOC0yNS4xDQoJCWM0LjUtOS43LDcuMy0yMC44LDguNC0zMi44bDAuMi0yLjRIODQxTDg0MC43LDM1Mi4xeiIvPg0KCTxwb2x5Z29uIGNsYXNzPSJzdDQiIHBvaW50cz0iOTQ4LjYsMzg4LjEgOTQ4LjYsMjMzLjkgOTAyLjQsMjMzLjkgOTAyLjQsNDI3LjUgMTA0MC45LDQyNy41IDEwNDAuOSwzODguMSAJIi8+DQo8L2c+DQo8L3N2Zz4NCg==" 
alt="opencl"  height="30"/>

</div>

Edge detection methods used

- Canny edge detection
- Difference of Gaussian
