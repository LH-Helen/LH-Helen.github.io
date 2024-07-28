---
title:  "「Project」Github+jekyll+Minimal Mistakes创建自己的博客"
mathjax: true
key: site_blog
toc: true
toc_sticky: true
category: [work, project]


---

<span id='head'></span>

# Minimal Mistakes 主题探索及个性化修改

[Minimal Mistakes 官方文档](https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/)

## 风格修改

### 修改字体

字体会随着网页大小的变化而变化，原作者设置了三个变化点。

_sass\minimal-mistakes\\\_reset.scss

```scss
html {
  /* apply a natural box layout model to all elements */
  box-sizing: border-box;
  background-color: $background-color;
  font-size: 16px;

  @include breakpoint($medium) {
    font-size: 18px;
  }
  // 删掉
  //@include breakpoint($large) {
  //  font-size: 20px;
  //}
  //
  //@include breakpoint($x-large) {
  //  font-size: 22px;
  //}

  -webkit-text-size-adjust: 100%;
  -ms-text-size-adjust: 100%;
}
```



## 问题：

### webrick

报错：

```
warning: webrick was loaded from the standard library, but is not part of the default gems since Ruby 3.0.0. Add webrick to your Gemfile or gemspec. Also contact author of jekyll-3.9.5 to add webrick into its gemspec.
```

解决：

```
bundle add webtrick
```



> 参考：[[Jekyll 运行的时候提示错误 cannot load such file -- webrick (LoadError)](https://www.cnblogs.com/huyuchengus/p/15473035.html)](https://www.cnblogs.com/huyuchengus/p/15473035.html)