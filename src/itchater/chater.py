import itchat

texts = []
@itchat.msg_register(itchat.content.TEXT)
def text_reply(msg):
    msg_text = msg.text
    texts.append(msg_text)
    if msg_text == "do":
        print(texts)
        texts.clear()
    return msg.text


itchat.auto_login()
itchat.run()