(() => {
  const data_config = {
    'ann_rain_2019': {
      'centred': true,
      'colourscale': 'RdYlBu_r',
      'dp': 3,
      'title': 'Observed rainfall anomaly in 2019',
      'units': 'mm/day',
    },
    'ann_rain_change_mmday_2065': {
      'centred': true,
      'colourscale': 'BrBG',
      'dp': 3,
      'title': 'Annual mean change in rainfall',
      'units': 'mm/day',
    },
    'ann_warming_oC_2065': {
      'colourscale': 'YlOrRd',
      'dp': 3,
      'title': 'Annual mean change in temperature',
      'units': '°C',
    },
    'drought_day_change_2065': {
      'colourscale': 'YlOrRd',
      'dp': 3,
      'title': 'Change in number of days experiencing drought',
      'units': 'days',
    },
    'max5day_rain_mm_change_2065': {
      'centred': true,
      'colourscale': 'BrBG',
      'dp': 3,
      'title': 'Change in amount of rain falling over wettest 5 days of the year',
      'units': 'mm/day',
    },
    'number_heatwave_days_change_2065': {
      'colourscale': 'YlOrRd',
      'dp': 3,
      'title': 'Change in number of days classed as heatwaves',
      'units': 'days',
    },
    'warmest_dayofyear_change_oC_2065': {
      'colourscale': 'YlOrRd',
      'dp': 3,
      'title': 'Change in temperature of warmest day of year',
      'units': '°C',
    }
  };

  // https://colorbrewer2.org
  const colourscales = {
    'Viridis': (min, max) => chroma.scale('Viridis').domain([min, max]),
    'RdYlBu_r': (min, max) => chroma.scale('RdYlBu').domain([max, min]),
    'YlOrRd': (min, max) => chroma.scale('YlOrRd').domain([min, max]),
    'BrBG': (min, max) => chroma.scale('BrBG').domain([min, max]),
  }

  // Colourscales from https://open-innovations.org/projects/hexmaps/builder.html
  // const colourscales = {
  //   'Viridis': 'rgb(122,76,139) 0%, rgb(124,109,168) 12.5%, rgb(115,138,177) 25%, rgb(107,164,178) 37.5%, rgb(104,188,170) 50%, rgb(133,211,146) 62.5%, rgb(189,229,97) 75%, rgb(254,240,65) 87.5%, rgb(254,240,65) 100%',
  //   'ODI': 'rgb(114,46,165) 0%, rgb(230,0,124) 50%, rgb(249,188,38) 100%',
  //   'Heat': 'rgb(0,0,0) 0%, rgb(128,0,0) 25%, rgb(255,128,0) 50%, rgb(255,255,128) 75%, rgb(255,255,255) 100%',
  //   'IMD-low-high': 'rgb(8,64,129) 0%, rgb(8,104,172) 10%, rgb(43,140,190) 20%, rgb(78,179,211) 30%, rgb(123,204,196) 40%, rgb(168,221,181) 50%, rgb(204,235,197) 60%, rgb(224,243,219) 70%, rgb(238,252,217) 80%, rgb(251,252,244) 90%, rgb(251,252,244) 100%',
  //   'IMD-high-low': 'rgb(251,252,244) 0%, rgb(238,252,217) 10%, rgb(224,243,219) 20%, rgb(204,235,197) 30%, rgb(168,221,181) 40%, rgb(123,204,196) 50%, rgb(78,179,211) 60%, rgb(43,140,190) 70%, rgb(8,104,172) 80%, rgb(8,64,129) 90%, rgb(8,64,129) 100%',
  //   'Planck': 'rgb(0,0,255) 0%, rgb(0,112,255) 16.666%, rgb(0,221,255) 33.3333%, rgb(255,237,217) 50%, rgb(255,180,0) 66.666%, rgb(255,75,0) 100%',
  //   'EPC': '#ef1c3a 1%, #ef1c3a 20.5%, #f78221 20.5%, #f78221 38.5%, #f9ac64 38.5%, #f9ac64 54.5%, #ffcc00 54.5%, #ffcc00 68.5%, #8cc63f 68.5%, #8cc63f 80.5%, #1bb35b 80.5%, #1bb35b 91.5%, #00855a 91.5%, #00855a 100%',
  //   'Plasma': 'rgb(12,7,134) 0%, rgb(82,1,163) 12.5%, rgb(137,8,165) 25%, rgb(184,50,137) 37.5%, rgb(218,90,104) 50%, rgb(243,135,72) 62.5%, rgb(253,187,43) 75%, rgb(239,248,33) 87.5%',
  //   'Referendum': '#4BACC6 0%, #B6DDE8 50%, #FFF380 50%, #FFFF00 100%',
  //   'Leodis': '#2254F4 0%, #F9BC26 50%, #ffffff 100%',
  //   'Longside': '#801638 0%, #addde6 100%',
  //   'Black': '#000000 0%, #000000 100%'
  // };

  const hex = new ODI.hexmap(
    document.querySelector('.hexmap__outer'),
    {
      'hexjson': 'resources/climate_projections.hexjson',
      'ready': () => {
        // Build dropdown options
        const hexmap_select = document.querySelector('.hexmap__select');
        for (const key in data_config) {
          const hexmap_opt = document.createElement('option');
          hexmap_opt.innerText = getLabel(key, data_config[key]);
          hexmap_opt.value = key;
          hexmap_select.appendChild(hexmap_opt);
        }

        // Load dataset from query parameter
        const params = new URLSearchParams(window.location.search);
        let active_key = params.get('data');
        if (!active_key || !(active_key in data_config)) {
          active_key = hexmap_select.value;
        }

        // Store some useful attrs for later
        hex.extra = {
          'activeKey': active_key,
          'colourbar': document.querySelector('.hexmap__colourbar')
        };

        // Update hexmap & add auto-update to select change
        hexmap_select.value = active_key;
        updateHexmap(hex, data_config, colourscales, active_key)
        hexmap_select.addEventListener('change', e => {
          updateHexmap(hex, data_config, colourscales, e.target.value);
        });
      }
    }
  );

  // Tooltip
  hex.on('mouseover', e => {
    const svg = e.data.hexmap.el;
    const hex = e.target;
    let tip = svg.querySelector('.tooltip');
    if (!tip) {
      tip = document.createElement('div');
      tip.classList.add('tooltip');
      svg.appendChild(tip);
    }

    const active_key = e.data.hexmap.extra.activeKey;
    const region_name = e.data.data.n;
    let region_contact = '';
    if ('mp_name' in e.data.data) {
      region_contact =  'MP: ' + e.data.data.mp_name;
      if ('first_party' in e.data.data) {
        region_contact += ' (' + e.data.data.first_party + ')';
      }
    }
    const region_value = getRegionValue(active_key, e.data.data, data_config);
    tip.innerHTML = [region_name, region_contact, region_value]
      .filter(item => item !== '')
      .join('<br>');

    const bb = hex.getBoundingClientRect();
    const bbo = svg.getBoundingClientRect();
    tip.style.left = Math.round(bb.left + bb.width/2 - bbo.left + svg.scrollLeft) + 'px';
    tip.style.top = Math.round(bb.top + bb.height/2 - bbo.top) + 'px';
  });

  hex.el.addEventListener('mouseleave', () => {
    const tooltip = hex.el.querySelector('.tooltip');
    tooltip && tooltip.remove();
  });


  function getLabel(key, conf) {
    if (!('title' in conf)) {
      return key;
    }

    return conf['title'] + ('units' in conf ? ' (' + conf['units'] + ')' : '');
  }

  function getRegionValue(key, data, conf) {
    if (!(key in data)) {
      return 'No data';
    }

    let value = data[key];
    let units = '';
    if (key in conf) {
      value = getRoundedValue(value, conf[key]['dp']);
      if ('units' in conf[key]) {
        units = ' ' + conf[key]['units'];
      }
    }

    return value + units;
  }

  function getRoundedValue(value, dp) {
    if (!dp) {
      return value;
    }

    return parseFloat(value).toFixed(dp);
  }

  function updateHexmap(obj, all_data_config, colourscales, key) {
    if (!(key in all_data_config)) {
      console.error('Error: ' + key + ' not found in config');
      return;
    }

    const config = all_data_config[key];
    const data = Object.values(obj.mapping.hexes)
      .map(item => item[key] || Number.NaN)
      .filter(item => !Number.isNaN(item));
    let vmin = Math.min(...data);
    let vmax = Math.max(...data);

    // Centre around 0
    if (config['centred']) {
      const abs_max = Math.max(Math.abs(vmin), Math.abs(vmax));
      vmin = abs_max * -1;
      vmax = abs_max;
    }

    // NB `colourscale_full` is a chroma.scale object (https://gka.github.io/chroma.js/#color-scales)
    const colourscale_full = colourscales[config.colourscale](vmin, vmax);
    obj.updateColours(r => colourscale_full(obj.mapping.hexes[r][key]));
    obj.extra.colourbar && updateColourbar(obj.extra.colourbar, colourscales, config, key, vmin, vmax);
    obj.extra.activeKey = key;

    // Update url
    const url = new URL(window.location)
    if (url.searchParams.get('data') != key) {
      url.searchParams.set('data', key);
      window.history.replaceState(null, '', url);
    }

    // Reset gridcells
    [...obj.el.querySelectorAll('.hex-cell.hover')].forEach(node => node.classList.remove('hover'));
  }

  function updateColourbar(el, colourscales, conf, key, vmin, vmax) {
    const inner = el.querySelector('.hexmap__colourbar__inner');
    inner.innerHTML = '';

    // NB `colourscale_norm` is a chroma.scale object (https://gka.github.io/chroma.js/#color-scales)
    const colourscale_norm = colourscales[conf.colourscale](0, 100);
    for (let i = 0; i < 100; i++) {
      const span = document.createElement('span');
      span.style.backgroundColor = colourscale_norm(i);
      inner.appendChild(span)
    }

    el.querySelector('.hexmap__colourbar__label').innerText = getLabel(key, conf);
    el.querySelector('.hexmap__colourbar__min').innerText = getRoundedValue(vmin, conf['dp']);
    el.querySelector('.hexmap__colourbar__max').innerText = getRoundedValue(vmax, conf['dp']);
  }
})();
