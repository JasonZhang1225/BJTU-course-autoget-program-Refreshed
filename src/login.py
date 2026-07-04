# coding=utf-8
import asyncio
import base64
import io
import json
import os
import re
import signal
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Union
from urllib import parse

import onnxruntime
import requests
import websockets
from bs4 import BeautifulSoup
from numpy import array, expand_dims, float32
from PIL import Image

onnxruntime.set_default_logger_severity(3)
warnings.filterwarnings("ignore")


class DdddOcr(object):
    def __init__(
        self,
        import_onnx_path: str = "",
    ):
        self.__graph_path = import_onnx_path
        # fmt: off
        self.__charset = [" ", "9", "5", "-", "7", "0", "2", "6", "1", "3", "x", "8", "=", "4", "+"]
        # fmt: on
        self.__resize = [-1, 64]

        self.__providers = ["CPUExecutionProvider"]
        self.__ort_session = onnxruntime.InferenceSession(
            self.__graph_path, providers=self.__providers
        )

    def classification(self, img):
        if not isinstance(img, (bytes)):
            raise TypeError("未知图片类型")

        image = Image.open(io.BytesIO(img))
        image = image.resize(
            (int(image.size[0] * (self.__resize[1] / image.size[1])), self.__resize[1]),
            Image.LANCZOS,
        ).convert("L")

        image = array(image).astype(float32)
        image = expand_dims(image, axis=0) / 255.0
        image = (image - 0.456) / 0.224

        ort_inputs = {"input1": array([image]).astype(float32)}
        ort_outs = self.__ort_session.run(None, ort_inputs)
        result = []

        last_item = 0
        for item in ort_outs[0][0]:
            if item == last_item:
                continue
            else:
                last_item = item
            if item != 0:
                result.append(self.__charset[item])
        return "".join(result)


def qwen_ocr(img: bytes, api_key: str, base_url: str) -> dict:
    """调用阿里云 Qwen OCR 识别验证码图片

    返回 {"success": True, "result": "xx"} 或 {"success": False, "message": "..."}
    base_url 示例: https://dashscope.aliyuncs.com/compatible-mode/v1
    代码会拼成端点在: {base_url}/chat/completions
    """
    b64 = base64.b64encode(img).decode()
    data_url = f"data:image/png;base64,{b64}"
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": "qwen-vl-ocr-latest",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": "图中是一个验证码，只输出图中的两个汉字。这两个字是一个语义连贯、现实中有意义的词语，不是两个无关的字。不要任何其他内容。",
                    },
                ],
            }
        ],
        "temperature": 0.01,
        "max_tokens": 10,
    }
    try:
        resp = requests.post(
            endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        if resp.status_code != 200:
            return {
                "success": False,
                "message": f"Qwen HTTP {resp.status_code}: {resp.text[:200]}",
            }
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # 清洗输出：只保留前两个中文字符
        chars = re.findall(r"[一-龥]", content)
        if len(chars) >= 2:
            return {"success": True, "result": "".join(chars[:2])}
        return {"success": False, "message": f"OCR 输出无效: {content[:50]}"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Qwen OCR 请求超时"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Qwen OCR 网络错误: {e}"}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {"success": False, "message": f"Qwen OCR 响应解析失败: {e}"}


def classify_failure(reason: str) -> str:
    """分类失败原因: captcha_error | quota_zero | other

    分类只用于前端配色，原样系统文本由调用方保留。
    """
    if "验证码" in reason:
        return "captcha_error"
    if "余量" in reason or "无课余量" in reason or "无余量" in reason:
        return "quota_zero"
    return "other"


@dataclass
class CourseConfig:
    """课程配置数据类"""

    qwen_api_key: str
    qwen_base_url: str  # 阿里云兼容 endpoint 的 base URL
    senior_check: bool
    monitor_only: bool
    course_list: List[str]
    username: str
    password: str
    model_path: str
    course_type: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CourseConfig":
        # 验证必需字段
        for key in [
            "qwenApiKey",
            "courseList",
            "seniorCheck",
            "monitorOnly",
            "username",
            "password",
            "modelPath",
            "courseType",
        ]:
            if key not in data:
                raise ValueError(f"数据中缺少{key}字段")

        return cls(
            qwen_api_key=data["qwenApiKey"],
            qwen_base_url=data.get(
                "qwenBaseUrl",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            course_list=[course.strip() for course in data["courseList"].split(",")],
            senior_check=data["seniorCheck"],
            monitor_only=data["monitorOnly"],
            username=data["username"],
            password=data["password"],
            model_path=data["modelPath"],
            course_type=data["courseType"],
        )


class CourseGrabber:
    """抢课核心类"""

    BASE_HEADERS = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    def __init__(self, config: CourseConfig):
        self.config = config
        self.session: requests.Session = None
        self.running = True
        self.username = None
        self.ocr = DdddOcr(
            import_onnx_path=self.config.model_path,
        )
        self.cookie = None
        self.session = requests.Session()
        self.last_quotas: Dict[str, int] = {}
        self.manual_captcha_queue: asyncio.Queue = asyncio.Queue()

    async def login(self, send) -> None:
        """登录获取会话。send 是一个 async 回调，用于把消息发给前端。
        不用 async generator，避免耗尽/清理时在某些环境下卡死。"""
        async def emit(msg):
            await send(msg)
        try:
            self.session.headers.update(self.BASE_HEADERS)

            await emit({"command": "登录", "std": "正在获取验证码..."})
            response = await asyncio.to_thread(self._get_initial_page, self.session)

            login_info = await asyncio.to_thread(
                self._extract_login_info, response.text
            )

            captcha_result = await asyncio.to_thread(
                self._handle_captcha, self.session, login_info["captcha_id"]
            )
            await emit(captcha_result)

            await emit({"command": "登录", "std": "正在登录..."})
            response = await asyncio.to_thread(
                self._do_login, self.session, login_info, captcha_result["result"]
            )

            await emit({"command": "登录", "std": "正在获取用户信息..."})
            await asyncio.to_thread(self._handle_redirects, self.session, response)

            self.username = await asyncio.to_thread(self._get_username, self.session)
            await emit({"command": "登录", "std": f"{self.username}, 登录成功!"})
            self.cookie = self.session.cookies.get_dict()
            await emit({"command": "登录", "std": "登录成功"})
        except requests.exceptions.RequestException as e:
            await emit({"command": "error", "error": f"登录失败: 网络请求失败: {str(e)}"})
            return
        except Exception as e:
            # 遍历到最内层栈帧，给出真正出错的函数名和行号
            tb = e.__traceback__
            deepest = tb
            while tb is not None:
                deepest = tb
                tb = tb.tb_next
            inner_func = deepest.tb_frame.f_code.co_name
            inner_line = deepest.tb_lineno
            await emit({
                "command": "error",
                "error": (
                    f"登录失败: {str(e)} [出错函数: {inner_func}() 第{inner_line}行]\n"
                    f"{traceback.format_exc()}"
                ),
            })
            return
            return

    def _get_initial_page(self, session: requests.Session) -> requests.Response:
        """获取初始登录页面"""
        response = session.get(
            "https://mis.bjtu.edu.cn/auth/sso/?next=/", allow_redirects=False
        )
        url = response.headers.get("Location")
        response = session.get(url, allow_redirects=False)
        url = "https://cas.bjtu.edu.cn" + response.headers.get("Location")
        return session.get(url, allow_redirects=False)

    def _extract_login_info(self, text: str) -> Dict[str, str]:
        """提取登录所需信息"""
        soup = BeautifulSoup(text, "html.parser")
        captcha_img = soup.find("img", class_="captcha")
        captcha_id = captcha_img["src"].split("/")[-2]
        csrf_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
        csrfmiddlewaretoken = csrf_input["value"]

        next_input = soup.find("input", {"name": "next"})
        next_url = next_input["value"].replace("&amp;", "&")

        return {
            "captcha_id": captcha_id,
            "csrfmiddlewaretoken": csrfmiddlewaretoken,
            "next_url": next_url,
        }

    def _handle_captcha(
        self, session: requests.Session, captcha_id: str
    ) -> Dict[str, str]:
        """处理验证码"""
        captcha_img_url = f"https://cas.bjtu.edu.cn/image/{captcha_id}"
        captcha_img = session.get(captcha_img_url).content

        try:
            start_time = time.time()
            expression = self.ocr.classification(captcha_img)

            expression = (
                expression.replace("x", "*")  # 将乘号x转换为*
                .replace("×", "*")  # 处理全角乘号
                .replace("=", "")  # 移除等号
                .strip()  # 移除空白
            )

            result = eval(expression, {"__builtins__": {}}, {})

            # 确保结果为整数
            if isinstance(result, float):
                result = int(result)
            process_time = round((time.time() - start_time) * 1000)

            return {
                "command": "captcha-image",
                "image": "data:image/png;base64,"
                + base64.b64encode(captcha_img).decode(),
                "result": result,
                "process_time": f"{process_time}ms",
            }
        except Exception as e:
            raise Exception(f"验证码计算失败: {str(e)}")

    def _do_login(
        self, session: requests.Session, login_info: Dict[str, str], captcha_result: str
    ) -> requests.Response:
        """执行登录请求"""
        url = f"https://cas.bjtu.edu.cn/auth/login/?next={login_info['next_url']}"
        payload = {
            "next": login_info["next_url"],
            "csrfmiddlewaretoken": login_info["csrfmiddlewaretoken"],
            "loginname": self.config.username,
            "password": self.config.password,
            "captcha_0": login_info["captcha_id"],
            "captcha_1": captcha_result,
        }

        session.headers.update(
            {
                "authority": "cas.bjtu.edu.cn",
                "content-type": "application/x-www-form-urlencoded",
                "origin": "https://cas.bjtu.edu.cn",
                "referer": f"https://cas.bjtu.edu.cn/auth/login/?next={parse.quote(login_info['next_url'])}",
            }
        )

        return session.post(url, data=payload, allow_redirects=False)

    def _handle_redirects(self, session: requests.Session, response: requests.Response):
        """处理登录后的重定向"""
        url = "https://cas.bjtu.edu.cn" + response.headers.get("Location")
        response = session.get(url, allow_redirects=False)

        session.headers.update({"authority": "mis.bjtu.edu.cn"})
        url = response.headers.get("Location")
        session.get(url, allow_redirects=False)

        response = session.get("https://mis.bjtu.edu.cn/module/module/10/")
        forms = re.findall(r"<form action=\"(.*?)\"", response.text)
        if not forms:
            snippet = response.text[:300]
            raise Exception(
                f"登录后重定向失败：在 mis.bjtu.edu.cn/module/module/10/ 页面未找到 <form> "
                f"(HTTP {response.status_code})，可能未真正登录成功。页面片段: {snippet!r}"
            )
        url = forms[0]

        session.headers.update(
            {
                "authority": "aa.bjtu.edu.cn",
                "referer": "https://mis.bjtu.edu.cn/",
            }
        )
        session.get(url, allow_redirects=False)

    def _get_username(self, session: requests.Session) -> str:
        """获取用户名（仅用于日志展示，失败不应阻断登录/抢课主流程）"""
        url = "https://aa.bjtu.edu.cn/schoolcensus/schoolcensus/stucensuscard/"
        session.headers.update(
            {
                "authority": "aa.bjtu.edu.cn",
                "referer": "https://aa.bjtu.edu.cn/notice/item/",
            }
        )
        try:
            response = session.get(url, timeout=10)
            matches = re.findall("<small>欢迎您，</small>(.*)\n", response.text)
            if matches:
                return matches[0]
            # URL 已失效（HTTP 404 等）或页面结构变更：回退占位，不阻断主流程
            print(
                f"[警告] 获取用户名失败 (HTTP {response.status_code})，"
                f"URL 可能已失效: {url}。使用占位用户名继续。"
            )
        except requests.exceptions.RequestException as e:
            print(f"[警告] 获取用户名网络异常: {e}。使用占位用户名继续。")
        return "用户"

    async def grab_course(self) -> Dict[str, str]:
        try:
            try:
                async for message in self.fetch_and_handle_data():
                    if not self.running:
                        yield {"command": "stopped", "std": "抢课已停止"}
                        return
                    yield message
            except Exception as e:
                yield {"command": "error", "std": f"单次抢课失败: {str(e)}"}

        except Exception as e:
            yield {"command": "error", "std": f"抢课过程发生错误: {str(e)}"}
            return

    async def submit_course(self, course_id: str, course_name: str = ""):
        """提交选课请求（Qwen OCR + 手动竞速；验证码错误时刷新重试一次）

        course_name 用于 grab-success / grab-failed 消息，让前端定位看板行。
        """
        BASE_HEADERS = {
            "accept": "*/*",
            "origin": "https://aa.bjtu.edu.cn",
            "referer": "https://aa.bjtu.edu.cn/course_selection/courseselecttask/selects/",
            "authority": "aa.bjtu.edu.cn",
            "x-requested-with": "XMLHttpRequest",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        }
        for attempt in range(2):
            # 每次重试都重置 headers，避免上一次 POST 留下的 content-type 污染 GET 请求
            self.session.headers.update(BASE_HEADERS)
            # 清理可能残留的 content-type（来自上一次 POST）
            self.session.headers.pop("content-type", None)

            try:
                # 刷新并获取验证码图片，校验返回内容，失败立即重试（最多 3 次）
                # 不做校验时服务器偶发返回 HTML 错误页/空字节，base64 编码后前端显示损坏图标
                # captcha/refresh 也可能返回空响应(non-JSON)，导致 json() 解析失败
                img = None
                key = None
                for _img_try in range(3):
                    refresh_resp = await asyncio.to_thread(
                        self.session.get,
                        "https://aa.bjtu.edu.cn/captcha/refresh/",
                    )
                    # 校验 refresh 响应：必须 200 + JSON + 含 key
                    try:
                        if (
                            refresh_resp.status_code != 200
                            or len(refresh_resp.content) == 0
                        ):
                            continue
                        key = refresh_resp.json()["key"]
                    except (ValueError, KeyError):
                        continue
                    captcha_img_url = f"https://aa.bjtu.edu.cn/captcha/image/{key}"
                    img_resp = await asyncio.to_thread(
                        self.session.get, captcha_img_url
                    )
                    if (
                        img_resp.status_code == 200
                        and img_resp.headers.get("Content-Type", "").startswith("image/")
                        and len(img_resp.content) > 0
                    ):
                        img = img_resp.content
                        break
                    # 校验失败，立即重试，不 sleep
                if img is None:
                    yield {
                        "command": "抢课",
                        "std": "验证码图片获取失败（服务器限流或会话失效），等待下一轮",
                    }
                    yield {
                        "command": "grab-failed",
                        "course_name": course_name,
                        "time": time.strftime("%H:%M:%S"),
                        "reason_raw": "验证码图片获取失败",
                        "reason_type": "other",
                    }
                    return
                b64 = "data:image/png;base64," + base64.b64encode(img).decode()
                yield {
                    "command": "captcha-image",
                    "image": b64,
                    "result": "识别中...",
                    "process_time": "",
                }

                start_time = time.time()
                result = await self.recognize_captcha_race(img)
                process_time = round((time.time() - start_time) * 1000)

                if not result["success"]:
                    yield {
                        "command": "captcha-image",
                        "image": b64,
                        "result": f"识别失败: {result.get('message', '未知错误')}",
                        "process_time": f"{process_time}ms",
                    }
                    if attempt == 0:
                        yield {
                            "command": "抢课",
                            "std": f"验证码识别失败，刷新重试: {result.get('message', '')}",
                        }
                        continue
                    yield {
                        "command": "抢课",
                        "error": f"验证码识别失败，放弃本轮: {result.get('message', '')}",
                    }
                    yield {
                        "command": "grab-failed",
                        "course_name": course_name,
                        "time": time.strftime("%H:%M:%S"),
                        "reason_raw": f"验证码识别失败: {result.get('message', '未知错误')}",
                        "reason_type": "captcha_error",
                    }
                    return

                answer = result["result"]
                yield {
                    "command": "captcha-image",
                    "image": b64,
                    "result": answer,
                    "process_time": f"{process_time}ms",
                    "source": result.get("source", ""),
                }

                payload = f"checkboxs={course_id}&hashkey={key}&answer={parse.quote(answer)}"
                self.session.headers.update(
                    {"content-type": "application/x-www-form-urlencoded"}
                )
                response = await asyncio.to_thread(
                    self.session.post,
                    "https://aa.bjtu.edu.cn/course_selection/courseselecttask/selects_action/?action=submit",
                    data=payload,
                    allow_redirects=False,
                )

                self.session.headers.update(
                    {
                        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                        "cache-control": "max-age=0",
                    }
                )

                response = await asyncio.to_thread(
                    self.session.get,
                    "https://aa.bjtu.edu.cn/course_selection/courseselecttask/selects/",
                )
                text = response.text
                messages = re.findall(r'message \+= "(.*)<br/>";', text)
                if messages:
                    full_msg = "\n".join(messages)  # 拼接全部，不再只取 messages[0]
                    if "验证码" in full_msg and attempt == 0:
                        yield {
                            "command": "抢课",
                            "std": "验证码错误，刷新重试...",
                        }
                        continue
                    yield {"command": "抢课", "std": full_msg}
                    if "成功" in full_msg:
                        yield {
                            "command": "grab-success",
                            "std": full_msg,
                            "course_name": course_name,
                            "time": time.strftime("%H:%M:%S"),
                        }
                    else:
                        yield {
                            "command": "grab-failed",
                            "course_name": course_name,
                            "time": time.strftime("%H:%M:%S"),
                            "reason_raw": full_msg,
                            "reason_type": classify_failure(full_msg),
                        }
                    return

            except requests.exceptions.RequestException as e:
                if attempt == 0:
                    yield {
                        "command": "抢课",
                        "std": f"网络请求失败，刷新重试: {str(e)}",
                    }
                    continue
                yield {
                    "command": "grab-failed",
                    "course_name": course_name,
                    "time": time.strftime("%H:%M:%S"),
                    "reason_raw": f"网络请求失败: {str(e)}",
                    "reason_type": "other",
                }
                exc_type, exc_value, exc_traceback = sys.exc_info()
                line_no = exc_traceback.tb_lineno
                func_name = exc_traceback.tb_frame.f_code.co_name
                raise Exception(f"{str(e), {line_no}, {func_name}}")

        yield {"command": "抢课", "std": "选课提交多次失败，等待下一轮轮询"}
        yield {
            "command": "grab-failed",
            "course_name": course_name,
            "time": time.strftime("%H:%M:%S"),
            "reason_raw": "选课提交多次失败，未收到系统提示",
            "reason_type": "other",
        }

    async def recognize_captcha_race(self, img: bytes) -> Dict[str, Any]:
        """竞速：Qwen OCR 自动识别 vs 用户手动输入，谁先出用谁

        返回格式：{"success": True, "result": "xx", "source": "手动输入"|"qwen-vl-ocr"}
                 或 {"success": False, "message": "..."}
        """
        # 丢弃前一次可能残留的手动输入
        while not self.manual_captcha_queue.empty():
            try:
                self.manual_captcha_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        qwen_task = asyncio.create_task(
            asyncio.to_thread(
                qwen_ocr, img, self.config.qwen_api_key, self.config.qwen_base_url
            )
        )
        manual_task = asyncio.create_task(self.manual_captcha_queue.get())

        deadline = time.monotonic() + 5
        pending = {qwen_task, manual_task}

        try:
            while pending and time.monotonic() < deadline:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=(deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if manual_task in done:
                    # 手动结果先到，使用手动
                    qwen_task.cancel()
                    return {
                        "success": True,
                        "result": manual_task.result(),
                        "source": "手动输入",
                    }

                if qwen_task in done:
                    qwen_result = qwen_task.result()
                    if qwen_result["success"]:
                        manual_task.cancel()
                        return {**qwen_result, "source": "qwen-vl-ocr"}
                    # Qwen 失败，继续等手动（如果有剩余时间）

            return {"success": False, "message": "识别超时或失败"}

        finally:
            for t in (qwen_task, manual_task):
                if not t.done():
                    t.cancel()

    async def fetch_and_handle_data(self) -> Generator[Dict[str, Any], None, None]:
        """获取并处理课程数据"""
        try:
            # 获取所有课程数据
            courses = self.get_available_courses()
            if not courses:
                yield {
                    "command": "选课",
                    "error": (
                        f"未获取到符合条件的课程（筛选关键词：{', '.join(self.config.course_list)}，"
                        f"类型：{self.config.course_type}）。请确认选课列表填写正确、"
                        f"且当前处于选课开放时段。"
                    ),
                }
                return

            # 生成状态信息
            yield {
                "command": "选课",
                "std": f"{self.username}, {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"待选课程：{', '.join(self.config.course_list)}",
            }

            # 处理已选课程
            finished_courses = []
            available_courses = []

            for course in courses:
                if course["id"] == "已选":
                    finished_courses.append(re.sub(r"\s+", " ", course["name"].strip()))
                else:
                    available_courses.append(course)

            # 发送余量快照给前端面板（不进日志，每轮都发）
            # 含可选课程(selected=False) + 已选课程(selected=True)
            # 已选课程也塞进快照，让用户第一轮就能看到"已选到"，且刚抢到的课在下一轮持续显示不闪现
            snapshot = [
                {
                    "name": re.sub(r"\s+", " ", c["name"].strip()),
                    "quota": int(c["number"]),
                    "selected": False,
                }
                for c in available_courses
            ]
            for fname in finished_courses:
                snapshot.append({"name": fname, "quota": 0, "selected": True})
            yield {"command": "quota-snapshot", "courses": snapshot}

            if len(available_courses) == 0:
                yield {"command": "success", "std": "抢课完成"}
                return

            # 输出已选课程
            if finished_courses:
                yield {
                    "command": "选课",
                    "std": f"已选课程：{', '.join(finished_courses)}",
                }

            # 输出可选课程信息
            course_info = ""
            for course in available_courses:
                course_name = re.sub(r"\s+", " ", course["name"].strip())
                course_info += (
                    f"{course['id']}, {course_name}, "
                    f"{course['teacher']}, {course['number']}\n"
                )
            if course_info:
                yield {"command": "选课", "std": course_info}

            # 检测余量变化：仅当上一轮为 0、本轮 > 0 时通知
            for course in available_courses:
                course_id = course["id"]
                curr = int(course["number"])
                prev = self.last_quotas.get(course_id)
                if prev is not None and prev == 0 and curr > 0:
                    yield {
                        "command": "quota-change",
                        "course": re.sub(r"\s+", " ", course["name"].strip()),
                        "quota": curr,
                    }
                self.last_quotas[course_id] = curr

            # 尝试选课
            for course in available_courses:
                if int(course["number"]) > 0:
                    course_name = re.sub(r"\s+", " ", course["name"].strip())
                    yield {
                        "command": "抢课",
                        "std": f"正在抢课，{course_name}, "
                        f"{course['teacher']}, {course['number']}",
                    }

                    # 仅监控模式：不发选课请求，只记录
                    if self.config.monitor_only:
                        yield {
                            "command": "监控",
                            "std": f"检测到余量 {course_name} {course['number']}，监控模式未提交",
                        }
                        continue

                    # 提交选课
                    async for result in self.submit_course(course["id"], course_name):
                        yield result

        except Exception as e:
            # 保留原始异常信息，不再用模糊的 (msg, {line}, {func}) 覆盖
            raise

    def get_available_courses(self) -> List[Dict[str, str]]:
        """获取可选课程列表"""
        try:
            if self.config.course_type == "required":
                url = (
                    "https://aa.bjtu.edu.cn/course_selection/courseselecttask/selects/"
                )
                table_id = 1
                name_column = 1
                number_column = 2
                teacher_column = 6
            elif self.config.course_type == "elective":
                url = "https://aa.bjtu.edu.cn/course_selection/courseselecttask/selects_action/?action=load&iframe=school&page=1&perpage=1000"
                table_id = 0
                name_column = 2
                number_column = 3
                teacher_column = 6
            else:
                raise Exception("课程类型不正确")
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            tables = soup.find_all("table")
            if not tables or len(tables) < table_id + 1:
                snippet = response.text[:400]
                raise Exception(
                    f"获取课程列表失败：{self.config.course_type} 课页面未找到表格 "
                    f"(URL={url}, HTTP {response.status_code}, "
                    f"找到 {len(tables)} 个 table，需要第 {table_id} 个)。"
                    f"可能未进入选课阶段或页面结构已变。页面片段: {snippet!r}"
                )
            courses = []

            for row in tables[table_id].find_all("tr"):
                cols = row.find_all("td")
                if not cols:
                    continue

                checkbox = cols[0].find("input")
                if not checkbox:
                    checkbox = cols[0].text.strip()
                else:
                    checkbox = checkbox["value"]

                course = {
                    "id": checkbox,
                    "name": cols[name_column].text.strip().replace("\n", " "),
                    "number": cols[number_column].text.strip(),
                    "teacher": cols[teacher_column].text.strip(),
                }

                # 检查是否符合选课条件
                if self._check_course_valid(course):
                    courses.append(course)

            return courses

        except Exception as e:
            # 保留原始异常信息，不再用模糊的 (msg, {line}, {func}) 覆盖
            raise

    def _check_course_valid(self, course: Dict[str, str]) -> bool:
        """检查课程是否符合选课条件"""
        flag = False
        for key in self.config.course_list:
            if key in course["name"]:
                flag = True

            if (
                (not self.config.senior_check)
                and ("高级" in course["name"])
                and ("高级" not in key)
            ):
                return False

        if not flag:
            return False

        return True

    def stop(self):
        """停止抢课"""
        self.running = False


class WebSocketServer:
    """WebSocket服务器类"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.grabbers: Dict[str, CourseGrabber] = {}
        self.grab_course_tasks: Dict[str, asyncio.Task] = {}

    async def stop(self):
        """停止服务器"""
        if self.server:
            self.is_running = False
            self.server.close()
            await self.server.wait_closed()
            print("\n服务器已安全关闭")

    async def handle_connection(self, websocket):
        """处理WebSocket连接"""
        client_id = str(id(websocket))

        try:
            async for message in websocket:
                input_data = json.loads(message)
                if input_data.get("command") == "stop":
                    if client_id in self.grabbers:
                        self.grabbers[client_id].stop()
                    await websocket.send(
                        json.dumps({"command": "finished", "std": "任务已停止"})
                    )
                elif input_data.get("command") == "manual-captcha":
                    # 手动输入验证码——放入队列供抢课线程竞速使用
                    answer = input_data.get("answer", "").strip()
                    if client_id in self.grabbers and re.match(
                        r"^[一-龥]{2}$", answer
                    ):
                        self.grabbers[client_id].manual_captcha_queue.put_nowait(answer)
                else:
                    if client_id in self.grab_course_tasks:
                        self.grab_course_tasks[client_id].cancel()
                    self.grab_course_tasks[client_id] = asyncio.create_task(
                        self.process_message(websocket, client_id, input_data)
                    )

        except Exception as e:
            print(f"WebSocket错误: {str(e)}")
        finally:
            if client_id in self.grabbers:
                self.grabbers[client_id].stop()
                del self.grabbers[client_id]
            if client_id in self.grab_course_tasks:
                self.grab_course_tasks[client_id].cancel()
                del self.grab_course_tasks[client_id]

    async def process_message(self, websocket, client_id, input_data):
        config = CourseConfig.from_dict(input_data)
        grabber = CourseGrabber(config)
        self.grabbers[client_id] = grabber
        grabber.running = True

        async def _send(msg):
            await websocket.send(json.dumps(msg))
        try:
            await grabber.login(_send)
        except Exception as e:
            print(f"登录阶段异常: {e}", flush=True)
        while grabber.running and not websocket.closed:
            async for result in grabber.grab_course():
                await websocket.send(json.dumps(result))
                if result["command"] in ["success", "error", "stopped"]:
                    await websocket.send(
                        json.dumps({"command": "finished", "std": "任务结束"})
                    )
                    return
            await asyncio.sleep(2)

        await websocket.send(json.dumps({"command": "success", "std": "任务结束"}))
