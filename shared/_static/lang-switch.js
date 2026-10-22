// 语言切换：在 /ja/ 和 /zh/ 的同一页面之间跳转。
// 按钮插进文章标题行的按钮组（和源码仓库 / 下载 / 全屏图标同一排），
// 不再用 position:fixed 浮在右上角，免得手机上遮住底下的按钮。
(function () {
  var m = location.pathname.match(/^(.*\/)(ja|zh)\/(.*)$/);
  if (!m) return;

  var root = m[1];
  var lang = m[2];
  var page = m[3] || "index.html";
  var other = lang === "ja" ? "zh" : "ja";

  function makeButton() {
    var a = document.createElement("a");
    a.href = root + other + "/" + page;
    a.textContent = other === "zh" ? "中文" : "日本語";
    a.title = other === "zh" ? "切换到中文版" : "日本語版へ";
    a.rel = "alternate";
    a.hreflang = other;
    a.className = "btn btn-sm lang-switch";
    return a;
  }

  function mount() {
    var a = makeButton();

    // 首选：文章标题行右侧的图标按钮组
    var row = document.querySelector(".article-header-buttons");
    if (row) {
      row.appendChild(a);
      return;
    }

    // 兜底：页面上没有那个按钮组时，退回原来的右上角浮动样式
    a.style.cssText = [
      "position:fixed", "top:10px", "right:14px", "z-index:9999",
      "padding:3px 10px", "border:1px solid currentColor", "border-radius:4px",
      "font-size:13px", "line-height:1.6", "text-decoration:none",
      "opacity:.75", "background:var(--pst-color-background, transparent)"
    ].join(";");
    document.body.appendChild(a);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();
