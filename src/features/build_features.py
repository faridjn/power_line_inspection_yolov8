def add_author_and_date():
    from IPython.core.display import HTML
    import datetime
    author = '<a href="https://github.com/faridjn">faridjn</a>'
    today = datetime.date.today().strftime('%B %d, %Y')
    text = f"Author: {author} <br> Date: {today}"
    display(HTML(text))