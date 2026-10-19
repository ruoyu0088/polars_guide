// 语言切换：在 /ja/ 和 /zh/ 的同一页面之间跳转
(function () {
  var m = location.pathname.match(/^(.*\/)(ja|zh)\/(.*)$/);
  if (!m) return;

  var root = m[1];
  var lang = m[2];
  var page = m[3] || "index.html";
  var other = lang === "ja" ? "zh" : "ja";

  var a = document.createElement("a");
  a.href = root + other + "/" + page;
  a.textContent = other === "zh" ? "中文" : "日本語";
  a.title = other === "zh" ? "切换到中文版" : "日本語版へ";
  a.rel = "alternate";
  a.hreflang = other;
  a.style.cssText = [
    "position:fixed", "top:10px", "right:14px", "z-index:9999",
    "padding:3px 10px", "border:1px solid currentColor", "border-radius:4px",
    "font-size:13px", "line-height:1.6", "text-decoration:none",
    "opacity:.75", "background:var(--pst-color-background, transparent)"
  ].join(";");

  document.addEventListener("DOMContentLoaded", function () {
    document.body.appendChild(a);
  });
})();
