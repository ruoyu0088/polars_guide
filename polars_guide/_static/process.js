document.addEventListener('DOMContentLoaded', () => {
    // ページ内のすべての <span> 要素を取得
    const spans = document.querySelectorAll('span.c1');
  
    // 各 <span> 要素を処理
    spans.forEach(span => {
      const textContent = span.textContent;
  
      // 丸数字 (❶～❾) に一致する場合
      const match = textContent.match(/#(❶|❷|❸|❹|❺|❻|❼|❽|❾)/);
      if (match) {
        // `#`と丸数字を分割し、新しいHTMLを設定
        span.innerHTML = `<span style="opacity: 0;">#</span><span style="font-size: 200%; color:#C76E00;">${match[1]}</span>`;
      }
    });


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
});