
**Notes on scraping with python**
**Author: Ocsphy**
### 目录
[TOC]

### HTTP和HTTPS

#### HTTP和网络传输协议
- HTTP是Hyper Text Transfer Protocol（超文本传输协议）的缩写。它的发展是万维网协会（World Wide Web Consortium）和Internet工作小组IETF（Internet Engineering Task Force）合作的结果，（他们）最终发布了一系列的RFC，RFC 1945定义了HTTP/1.0版本。其中最著名的就是RFC 2616。RFC 2616定义了今天普遍使用的一个版本——HTTP 1.1。

- HTTP是一个应用层协议，由 **请求(Resquest)** 和 **响应(response)** 构成，是一个标准的客户端服务器模型。HTTP是一个无状态的协议。
-HTTP协议通常承载于TCP协议之上，有时也承载于TLS或SSL协议层之上，这个时候，就成了我们常说的HTTPS。如下图所示：
<div align="center"><img src="pic\TCP-IP.jpg"></div>

- HTTPS的全称是Hyper Text Transfer Protocol over Secure Socket Layer，是以安全为目标的HTTP通道，简单讲是HTTP的安全版，即HTTP下加入SSL层，简称为HTTPS。
HTTPS的安全基础是SSL，因此通过它传的内容都是经过SSL加密的，它的主要作用可以分为两种。
  * 建立一个信息安全通道来保证数据传输的安全。
  * 确认网站的真实性，凡是使用了HTTPS的网站，都可以通过点击浏览器地址栏的锁头标志来查看网站认证之后的真实信息，也可以通过CA机构颁发的安全签章来查询。

#### HTTP请求和相应过程
- 我们在浏览器中输入一个URL，回车之后便会在浏览器中观察到页面内容。实际上，这个过程是浏览器向网站所在的服务器发送了一个请求，网站服务器接收到这个请求后进行处理和解析，然后返回对应的响应，接着传回给浏览器。响应里包含了页面的源代码等内容，浏览器再对其进行解析，便将网页呈现了出来。
<div align="center"><img src="pic\requestProcess.jpg"></div>

 ##### 1. 请求过程 

- 以淘宝为例，打开淘宝后，请求的内容如下所示：
<div align="center"><img src="pic\baidu.jpg"></div>

- 一般而言第一个返回的Response是一个document类型的文件，是网页的主代码，也是我们主要要获取和解析的部分；然后其他部分则是这个主的response的一个超链接、视频、图片、音频等的Response。

##### 2. 请求内容

- **请求方式（Request Method）**  
  常用的有GET和POST两种请求的方式，他们的区别如下：
  <div align="center"><img src="pic\get_post.jpg"></div>
  打开详细信息，在General部分我们可以看到请求的方法类型：
    <center class="half">
    <img src="pic\get.jpg" width = 350>
    <img src="pic\post.jpg" width = 450>
  </center>

- **请求头（Request Header）**
  请求头包含了请求的很多重要信息：
  Content | Description
  --      | ------------------------
  Accept  | 请求报头域，用于指定客户端可接受哪些类型的信息。
  Accept-Language |指定客户端可接受的语言类型。
  Host| 用于指定请求资源的主机IP和端口号，其内容为请求URL的原始服务器或网关的位置。从HTTP 1.1版本开始，请求必须包含此内容。
  Cookie| 这是网站为了辨别用户进行会话跟踪而存储在用户本地的数据。它的主要功能是维持当前访问会话。Cookies里有信息标识了我们所对应的服务器的会话，每次浏览器在请求该站点的页面时，都会在请求头中加上Cookies并将其发送给服务器，服务器通过Cookies识别出是我们自己，并且查出当前状态是登录状态，所以返回结果就是登录之后才能看到的网页内容。
  Referer | 此内容用来标识这个请求是从哪个页面发过来的，服务器可以拿到这一信息并做相应的处理，如作来源统计、防盗链处理等。
  User-Agent| 简称UA，它是一个特殊的字符串头，可以使服务器识别客户使用的操作系统及版本、浏览器及版本等信息。在做爬虫时加上此信息，可以伪装为浏览器；如果不加，很可能会被识别出为爬虫。
  Content-Type| 也叫互联网媒体类型（Internet Media Type）或者MIME类型，在HTTP协议消息头中，它用来表示具体请求中的媒体类型信息。例如，text/html代表HTML格式，image/gif代表GIF图片，application/json代表JSON类型，
  <center class="half">
    <img src="pic\header.jpg" width="400"/>
    <img src="pic\post_header.jpg" width="300"/>
  </center>

- **表单（Form）**
  在进行POST请求的时候（比如登录），我们需要向服务器提交除了header以外的信息，即**请求体**。常见的就是**表单**。在提交表单的时候，我们**必须**在请求头的**Content-Type**中指定我们要上传的数据类型：
  Content-Type | 数据类型
  --|--
  application/x-www-form-urlencoded|表单数据
  multipart/form-data|表单文件上传
  application/json|序列化JSON数据
  text/xml|XML数据

##### 3. 相应内容
- **相应状态**
  有多种相应状态码，比如200表示成功，301表示跳转，404表示找不到网页，502表示服务器错误等。

- **响应头（Response Header）**
  返回多种响应的信息：
  Content | Description
  --      | ------------------------
  Date| 标识响应产生的时间。
  Last-Modified|指定资源的最后修改时间。
  Content-Encoding|指定响应内容的编码。
  Server|包含服务器的信息，比如名称、版本号等。
  Content-Type|文档类型，指定返回的数据类型是什么，如text/html代表返回HTML文档，application/x-javascript则代表返回JavaScript文件，image/jpeg则代表返回图片。
  Set-Cookie|设置Cookies。响应头中的Set-Cookie告诉浏览器需要将此内容放在Cookies中，下次请求携带Cookies请求。
  Expires|指定响应的过期时间，可以使代理服务器或浏览器将加载的内容更新到缓存中。如果再次访问时，就可以直接从缓存中加载，降低服务器负载，缩短加载时间。
  <div align="center"><img src="pic\response.jpg" height="400"></div>

- **响应体**
  最重要的部分，包含了请求的资源内容，比如网页HTML，图片二进制数据等。


#### 爬虫的学习路径
<div align="center"><img src="pic\learn_path.jpg"></div>

---------------

### urllib库
urllib库是python内置的HTTP请求库，内置四个模块：
module | description
--|--
urllib.request|请求模块
urllib.error|异常处理模块 
urllib.parse|url解析模块 
urllib.robotparser|rebots.txt解析模块 

#### 1.urllib.request
##### urlopen()
- urllib.request.urlopen(url, data=None, [timeout, ]*, cafile=None, capath=None, cadefault=False, context=None)
  <table>
  <tr>
  <th>Input</th> <th>Type</th><th>Description</th>
  </tr>
  <tr>
  <td>url</td> <td>str</td><td>URL路径</td>
  </tr>
  <tr>
  <td>data</td> <td>bytes, file-like objects, and iterables</td> <td>请求时附加的数据，如header、form等</td> 
  </tr>
  <tr>
  <td>timeout</td> <td>int, float</td> <td>超时设置,即请求应当在t秒内得到响应，否则抛出异常</td> 
  </tr>
  <tr>
  <th>Output</th> <th>Type</th><th>Description</th>
  </tr>
  <tr>
  <td>HTTPResponse</td> <td>object</td> <td>返回一个对象</td> 
  </tr>
  </table>

- Get请求

  ```python
  import urllib.request
  request = urllib.request.urlopen("http://www.taobao.com")
  print(request.read().decode('utf-8'))  # decode('utf-8')转成字符串
  ```

  输出结果如下所示：
  <div align="center"><img src="pic\taobao.jpg"></div>

- POST请求

  ```python
  import urllib.request
  import urllib.parse
  data = bytes(urllib.parse.urlencode({"word":"hello"}),encoding = 'utf-8')
  request = urllib.request.urlopen("http://httpbin.org/post",data = data)
  # http://httpbin.org/post 是一个http测试网站
  print(request.read())
  ```

  输出结果如下所示：
  <div align="center"><img src="pic\urllib.post.jpg"></div>

- Timeout
  ```python
  import urllib.request
  request = urllib.request.urlopen("http://httpbin.org/get",timeout=1)
  print(request.read())
  ```

  输出结果如下所示：
  <div align="center"><img src="pic\urllib_timeout.jpg"></div>

  ```python
  import socket
  import urllib.request
  import urllib.error
  try:
      response = urllib.request.urlopen("http://httpbin.org/get",timeout = 0.1)
  except urllib.error.URLError as e :
      if isinstance(e.reason,socket.timeout):
          print("TIMEOUT")
  #out: TIMEOUT
  ```
 - **HTTPResponse**
 urllib.request.urlopen返回的结果是一个对象，这个对象有如下方法：

  <table>
  <tr>
  <th>Method</th> <th>Description</th>
  </tr>
  <tr>
  <td>read</td> <td>返回读取的响应体内容，bytes</td>
  </tr>
  <tr>
  <td>status</td> <td>返回状态码</td> 
  </tr>
  <tr>
  <td>getheaders</td> <td>返回响应头信息，list</td>
  </tr>
  <tr>
  <td>getheader(header)</td> <td>返回指定的响应头信息，str</td>
  </tr>
  <tr>
  <td>geturl</td> <td>返回url信息，str</td>
  </tr>
  </table>
  
  ```python
  import urllib.request
  response = urllib.request.urlopen("http://www.python.org")
  print(type(response))
  print(response.geturl())
  print(response.status)
  print(response.getheaders())
  print(response.getheader('Content-Type'))

  #out:
  # <class 'http.client.HTTPResponse'>
  # https://www.python.org/
  # 200
  # [('Server', 'nginx'), ('Content-Type', 'text/html; charset=utf-8'), ('X-Frame-Options', 'DENY'), ('Via', '1.1 vegur'), ('Via', '1.1 varnish'), ('Content-Length', '49433'), ('Accept-Ranges', 'bytes'), ('Date', 'Wed, 22 May 2019 11:49:21 GMT'), ('Via', '1.1 varnish'), ('Age', '3541'), ('Connection', 'close'), ('X-Served-By', 'cache-iad2146-IAD, cache-hkg17921-HKG'), ('X-Cache', 'HIT, HIT'), ('X-Cache-Hits', '1, 1008'), ('X-Timer', 'S1558525761.251910,VS0,VE0'), ('Vary', 'Cookie'), ('Strict-Transport-Security', 'max-age=63072000; includeSubDomains')]
  # text/html; charset=utf-8
  ```

##### Request()
[urlopen()](#urlopen)可以简单的获取网页内容，但是有一个问题，就是不能提交header数据；Request()可以解决这个问题：
- urllib.request.Request(url, data=None, headers={}, origin_req_host=None, unverifiable=False, method=None)

  <table>
  <tr>
  <th>Input</th> <th>Type</th><th>Description</th>
  </tr>
  <tr>
  <td>url</td> <td>str</td><td>URL路径</td>
  </tr>
  <tr>
  <td>data</td> <td>bytes, file-like objects, and iterables</td> <td>请求时附加的数据，如header、form等</td> 
  </tr>
  <tr>
  <td>header</td> <td>dictionary</td> <td>请求头</td> 
  </tr>
  <tr>
  <td>method</td> <td>str</td> <td>请求方式</td> 
  </tr>
  <tr>
  <th>Output</th> <th>Type</th><th>Description</th>
  </tr>
  <tr>
  <td>HTTPResponse</td> <td>object</td> <td>返回一个对象</td> 
  </tr>
  </table>

  ```python
  from urllib import request, parse
  url = "http://httpbin.org/post"
  header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Host": "httpbin.org"}
  data_dic = {"name": "Ocsphy"}
  data = bytes(parse.urlencode(data_dic), encoding='utf-8')
  req = request.Request(url=url, data=data, headers=header, method="POST")
  '''
  以下方式也是可以添加请求头的
  req = request.Request(url=url, data=data, method="POST")
  req.add_header("Host": "httpbin.org")
  '''
  response =request.urlopen(req)
  print(response.read().decode('utf-8'))
  ```

  输出结果如下，左边为用Request提交headers的结果，右边是用url的结果。可以看到大体内容差不多，就Agent发生了变化，这就导致了在反爬虫里面，urlopen更容易被发现。
    <center class="half">
    <img src="pic\Request_agent.jpg" width="500"/>
    <img src="pic\urlopen_agent.jpg" width="300"/>
  </center>

##### Handler
在request库中，还提供了很多Handler对象供处理代理、Cookies之类的工作
- 代理
  Http代理是指通过代理服务器访问目标服务器，可以使得在爬虫的时候用户IP发生变化从而防止反爬虫机制对IP的封锁。基本用法如下：
  ```python
  import urllib.request
  proxy = urllib.reques.ProxyHandler({
    "http": ****
    "https": ****
  })
  opener = urllib.request.build_opener(proxy)
  response = opener.open(*url*)
  ```


