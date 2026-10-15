迪士尼乐园内部知识与工具操作手册
（适用于一线员工及客服中心，版本 2025-07-28）

一、CRM 系统（Salesforce）操作指南

1. 登录与入口
• 内网地址：[https://crm.disney.cn（上海）、https://crm.disney.com（全球）](https://crm.disney.xn--cn%28%29https-yj3hh33zk21d/crm.disney.com%EF%BC%88%E5%85%A8%E7%90%83%EF%BC%89)
• 账号：工号＋一次性验证码（OTP）
• 常用模块：
├─「Cases」：投诉/咨询工单
├─「Guest Profile」：客史、偏好、年卡信息
└─「Magic Moments」：授权补偿券发放
2. 新建工单（Case）
① 点击「New Case」→ 选择园区/部门 → 输入门票编号或手机号自动匹配客人
② 「Case Type」下拉选择：门票、酒店、商品、安全、系统故障等
③ 「Priority」等级：P1（安全/舆情）、P2（现场投诉）、P3（事后咨询）
④ 填写「Subject」+「Description」→ 保存后系统自动生成 Case ID
⑤ 关联「Asset」字段：如尊享卡、客房号、演出场次，方便后续追踪
3. 常用快捷键
• Alt+S：保存并分派给下一环节
• Ctrl+Shift+C：复制 Case URL 到剪贴板
• @mention：@宾客关系部 直接提醒升级
4. 客史查询
• 在「Guest Profile」输入手机号或邮箱 → 查看「Visit History」「消费记录」「黑名单标记」
• 若发现「VIP Flag」= Gold/Platinum，立即通知当班主管
5. 常见问题及解决
• 登录超时：清除浏览器缓存→重新 OTP
• 工单打不开：检查是否被其他同事锁定，点击「Unlock」
• 字段缺失：刷新页面或切换「Lightning/Classic」视图

二、内部预订 & 查询系统

1. 门票预订（上海示例）
① 登录内部预订系统：[http://internal.shdr.com](http://internal.shdr.com/)
② 菜单「Ticketing」→「Create Order」→ 选择票种、日期、数量
③ 关联 CRM 客史：输入手机号自动带出姓名、年卡级别
④ 生成订单号（格式：GAL-YYYYMMDD-6位随机） → 点击「Issue」→ 短信/邮件自动推送二维码
2. 酒店预订
① 「Hotel」→ 选酒店（乐园酒店/玩具总动员酒店）→ 选房型 → 选日期
② 支持「Hold Room」功能：为投诉升级客人留房 24 h
③ 确认后生成确认号（SHDR-H-8位数字）→ 同步至 PMS（Opera）
3. 查询 & 报表
• 实时客流：「Dashboard」→「Live Occupancy」
• 退款列表：「Finance」→「Refund Queue」→ 勾选→「Batch Refund」

三、退款系统操作步骤

| **场景** | **系统路径** | **必填字段** | **自动规则** | **到账时间** |
| --- | --- | --- | --- | --- |
| 门票未使用 | Ticketing → Refund → 选择订单 → 原因「Unused」 | 订单号、退款路径 | 0 手续费 | 3-5 个工作日 |
| 天气闭园 | Dashboard → Force Majeure → 勾选日期 →「Auto Refund」 | 闭园公告编号 | 全额退 | T+1 |
| 酒店取消 | Hotel → Cancellation → 选择确认号 | 确认号、取消原因 | 按政策扣款 | 7-14 个工作日 |
| 尊享卡系统故障 | CRM → Magic Moments → 补偿券类型「FASTPASS」 | Case ID | 自动发放二维码 | 即时 |

四、常见问题速查表

| **问题** | **解决动作** | **备注** |
| --- | --- | --- |
| 内部预订系统白屏 | 换浏览器 Edge/Chrome 或清缓存 | 90% 解决 |
| 退款按钮灰色 | 检查订单状态是否为「Used」或「Checked-in」 | 已使用不可退 |
| CRM 无法搜索客人 | 尝试去掉国家区号或用邮箱搜索 | 国际手机号格式差异 |
| 年卡延期失败 | 在 CRM「Membership」→「Extend」输入延期天数 | 需主管二次审批 |

五、升级与求助
• 技术故障报修：内网 IM @IT-Support，附截图+Case ID
• 权限申请：填写「Access Request Form」，主管签字后 2 h 内开通
• 紧急电话：
├─ 上海：分机 8888（24 h）
├─ 全球：+1-407-939-HELP迪士尼乐园内部 CRM 系统（Salesforce）的登录入口为：

- **全球通用云端地址
**[https://login.salesforce.com](https://login.salesforce.com/)
（凭公司 SSO 或工号+一次性验证码登录）
- **上海迪士尼乐园专属子域（示例）
**[https://disneysh.my.salesforce.com](https://disneysh.my.salesforce.com/)
（仅供内部演职人员使用，需内网或 VPN 环境）
- **香港迪士尼乐园子域（示例）
**[https://disneyhk.my.salesforce.com](https://disneyhk.my.salesforce.com/)
（同样需内部账号与 MFA 验证）

以上地址均为 Salesforce 托管的云端登录页，外部游客无法访问