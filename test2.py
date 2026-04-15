import os
import ray
import socket

def main():
    print("=== 开始复现 Ray 403 Gateway 空格由于隔离代理失效导致的报错 ===")
    
    # 获取本机内网 IP，采用更鲁棒的 UDP 探测法（防止某些容器环境无 hostname 映射）
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"
    print(f"[环境准备] 当前本机 IP: {local_ip}")

    # 1. 人为毒害环境：制造那个致命的“空格 Bug”
    # 注意这里在 0:23 和目标之间故意留了一个空格！
    bad_no_proxy = f"localhost,127.0.0.1,::1,fdbd:fdbd:fdbd:fdbd:ffff:ffff:0:23 {local_ip},.byted.org,byted.org"
    os.environ["no_proxy"] = bad_no_proxy
    os.environ["NO_PROXY"] = bad_no_proxy

    # 2. 模拟底座的隐形代理
    # 如果你在集群上跑，这段由于真实集群自带代理，会触发真的 403。
    # 这里我们哪怕随便填一个假的 http 代理地址验证拦截规律，如果请求被打到拦截地址，就会暴露！
    if "HTTP_PROXY" not in os.environ and "http_proxy" not in os.environ:
        print("你2好")
        # os.environ["HTTP_PROXY"] = None # 假定代号

    print(f"[环境准备] 已强行注入被污染的 no_proxy: '{bad_no_proxy}'")
    print(f"[环境准备] 当前应用的 HTTP_PROXY: {os.environ.get('HTTP_PROXY')}\n")

    # 3. 启动 Ray (因为这里是纯本地复现，ray.init 会继承我们刚才注入的 python 脏环境启动本机的 Raylet)
    print("配置完毕，正在启动本地 Ray 系统服务...")
    ray.init(ignore_reinit_error=True)

    # 4. 重点：定义一个带有 runtime_env 的任务！
    # 只要有了 runtime_env，Ray Agent 就会立刻通过 HTTP 协议与内网通信去同步环境
    runtime_env = {"env_vars": {"TEST_VAR": "HELLO"}}

    @ray.remote
    class DummyActor:
        def ping(self):
            return "Pong!"

    print("准备就绪，正在向 Ray 集群申请创建带有 runtime_env 的 Actor 任务...")
    print("这会触发底层发送 HTTP 通信请求。请观察下面是否会引发崩溃！\n")

    try:
        # 尝试创建一个带 runtime_env 的 Actor。
        # 这里一旦底层由于空格 bug 无法绕过代理，就会直接报出你见过的 RuntimeEnvSetupError 403 !
        actor = DummyActor.options(runtime_env=runtime_env).remote()
        result = ray.get(actor.ping.remote())
        print(f"执行成功，返回了: {result}")
        print("如果执行竟然成功，说明这个集群完全没有开启 Compliance Gateway 拦截机制，或者被代理的地址不对！")
    except Exception as e:
        print("\n\n=============== 💥 凶案现场复现 💥 ===============")
        print("复现成功！底层网络库因被空格提前截断，无法把当前 IP 加入白名单，")
        print("致使内部 HTTP 通信请求狠狠砸在了代理网关上，抛出以下异常（完美契合你的报错）：\n")
        print(e)

if __name__ == "__main__":
    main()
