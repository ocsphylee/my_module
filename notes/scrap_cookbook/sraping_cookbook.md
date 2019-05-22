
**Notes on scraping with python**
**Author: Ocsphy**
[TOC]

### HTTP 和 HTTPS

#### HTTP和网络传输协议
- HTTP是Hyper Text Transfer Protocol（超文本传输协议）的缩写。它的发展是万维网协会（World Wide Web Consortium）和Internet工作小组IETF（Internet Engineering Task Force）合作的结果，（他们）最终发布了一系列的RFC，RFC 1945定义了HTTP/1.0版本。其中最著名的就是RFC 2616。RFC 2616定义了今天普遍使用的一个版本——HTTP 1.1。

- HTTP是一个应用层协议，由 **请求(Resquest)** 和 **响应(respond)** 构成，是一个标准的客户端服务器模型。HTTP是一个无状态的协议。
-HTTP协议通常承载于TCP协议之上，有时也承载于TLS或SSL协议层之上，这个时候，就成了我们常说的HTTPS。如下图所示：
<div align="center"><img src="pic\TCP-IP.jpg"></div>

- HTTPS的全称是Hyper Text Transfer Protocol over Secure Socket Layer，是以安全为目标的HTTP通道，简单讲是HTTP的安全版，即HTTP下加入SSL层，简称为HTTPS。
HTTPS的安全基础是SSL，因此通过它传的内容都是经过SSL加密的，它的主要作用可以分为两种。
  * 建立一个信息安全通道来保证数据传输的安全。
  * 确认网站的真实性，凡是使用了HTTPS的网站，都可以通过点击浏览器地址栏的锁头标志来查看网站认证之后的真实信息，也可以通过CA机构颁发的安全签章来查询。

#### HTTP请求过程
- 我们在浏览器中输入一个URL，回车之后便会在浏览器中观察到页面内容。实际上，这个过程是浏览器向网站所在的服务器发送了一个请求，网站服务器接收到这个请求后进行处理和解析，然后返回对应的响应，接着传回给浏览器。响应里包含了页面的源代码等内容，浏览器再对其进行解析，便将网页呈现了出来。
<div align="center"><img src="pic\requestProcess.jpg"></div>

 ##### 1. Google Chrome [检查] 请求过程
 - 以淘宝为例，打开淘宝后，请求的内容如下所示：
 <div align="center"><img src="pic\Chrome.png"></div>

Name |Status |Type |Initiator |Size |Time |Waterfall
--- | --- | --- | --- | --- | --- | --- |
请求的名称，一般会将URL的最后一部分内容当作名称 | 响应的状态码| 请求的文档类型 | 请求源,用来标记请求是由哪个对象或进程发起的 | 从服务器下载的文件和请求的资源大小。如果是从缓存中取得的资源，则该列会显示from cache |发起请求到获取响应所用的总时间 |网络请求的可视化瀑布流


  
