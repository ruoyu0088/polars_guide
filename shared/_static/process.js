function renderMath() {
    function processNode(node) {
        if (node.tagName === "PRE") return; // Skip <pre> elements

        if (node.nodeType === Node.TEXT_NODE) {
            let text = node.nodeValue;
            if (!text.trim()) return;

            let parts = [];
            let lastIndex = 0;

            // Match both inline ($...$) and display ($$...$$) math
            text.replace(/\$\$(.*?)\$\$|\$(.*?)\$/gs, (match, displayMath, inlineMath, offset) => {
                // Push normal text before match
                if (offset > lastIndex) {
                    parts.push(document.createTextNode(text.slice(lastIndex, offset)));
                }

                // Create math element
                let mathElement = document.createElement(displayMath ? "div" : "span");
                mathElement.className = "math";
                mathElement.textContent = displayMath ? `\\[${displayMath}\\]` : `\\(${inlineMath}\\)`;
                parts.push(mathElement);

                lastIndex = offset + match.length;
            });

            // Push any remaining normal text
            if (lastIndex < text.length) {
                parts.push(document.createTextNode(text.slice(lastIndex)));
            }

            // Replace only if changes were made
            if (parts.length > 0) {
                let parent = node.parentNode;
                parts.forEach((el) => parent.insertBefore(el, node));
                parent.removeChild(node);
            }
        } else {
            // Recursively process child nodes
            Array.from(node.childNodes).forEach(processNode);
        }
    }

    processNode(document.body);
    MathJax.typeset();
}

document.addEventListener("DOMContentLoaded", renderMath);



document.addEventListener('DOMContentLoaded', () => {
    // 放大注释里用作标记的圆数字（❶～❾）。两种写法都要处理：
    //   #❶                     —— 注释本身就是标记，隐藏 #，只放大圆数字
    //   # ❶ 前 8 个字节 = 1.0   —— 标记夹在注释正文里，只放大圆数字，其余文字原样保留
    const MARKER_RE = /[❶❷❸❹❺❻❼❽❾]/;
    const MARKER_ALL = /[❶❷❸❹❺❻❼❽❾]/g;
    const MARKER_STYLE = 'font-size: 200%; color:#C76E00;';

    function escapeHTML(s) {
      return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function enlargeMarkers(span) {
      const text = span.textContent;
      if (!MARKER_RE.test(text)) return;

      // 情形一：整段注释只有一个标记，形如 #❶ 或 # ❶
      const only = text.trim().match(/^#\s*([❶❷❸❹❺❻❼❽❾])$/);
      if (only) {
        span.innerHTML = '<span style="opacity: 0;">#</span>' +
          `<span style="${MARKER_STYLE}">${only[1]}</span>`;
        return;
      }

      // 情形二：标记出现在注释正文中间
      let html = '';
      let lastIndex = 0;
      text.replace(MARKER_ALL, (match, offset) => {
        html += escapeHTML(text.slice(lastIndex, offset)) +
          `<span style="${MARKER_STYLE}">${match}</span>`;
        lastIndex = offset + match.length;
        return match;
      });
      html += escapeHTML(text.slice(lastIndex));
      span.innerHTML = html;
    }

    document.querySelectorAll('span.c1').forEach(enlargeMarkers);


  const divs = document.querySelectorAll('div.cell_input.docutils.container');

  // Iterate through each div to check if it contains the code 'import panel as pn'
  divs.forEach(div => {
    if (div.textContent.includes('import panel as pn') || div.textContent.includes('import hvplot')) {
      console.log('Found the div containing "import panel as pn".');
      const nextDiv = div.nextElementSibling;
      if (nextDiv) {
        nextDiv.style.display = 'none'; // Hide the next div
      }      
    }
  });

  document.querySelectorAll(".plotly-graph-div").forEach(plotlyDiv => {
    let grandparent = plotlyDiv.parentElement?.parentElement;
    if (grandparent) {
        let elementBefore = grandparent.previousElementSibling;
        if (elementBefore) {
            elementBefore.style.display = "none";
        }
    }
});

renderMath();

});