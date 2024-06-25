(() => {
  const data_periods = {
    'obs': {
      'years': '1991-2020 vs 1961-1990'
      'title': 'Past changes'
    },
    'proj': {
      'years': '2021-2040 vs 1981-2010'
      'title': 'Near-term future'
    }
  };

  const data_vars = {
    'prdjf': {
      'colourscale': 'BrBG',
      'dp': 3,
      'title': 'Winter rainfall',
      'units': '%',
      'description': 'Winter is the wettest season across most of the UK. Not all the country has the same amount of rainfall, with more falling on the Western side of the country. Therefore it is common to show rainfall changes as percentages. Here winter is taken as December, January and February.',
      'periods': {
        'obs': {
          'min': -25,
          'max': 25
        },
        'proj': {
          'min': -10,
          'max': 10
        }
      }
    },
    'prjja': {
      'colourscale': 'BrBG',
      'dp': 3,
      'title': 'Summer rainfall',
      'units': '%',
      'description': 'Summer is drier than winter in many parts of the UK (although East Anglia is dry all year round). It is common to show rainfall changes as percentages, as they are often more interpretable than using mm/day. Here Summer is taken as June, July and August.',
      'periods': {
        'obs': {
          'min': -25,
          'max': 25
        },
        'proj': {
          'min': -10,
          'max': 10
        }
      }
    },
    'rx5day': {
      'colourscale': 'YlGnBu',
      'dp': 3,
      'title': 'Rain on wettest 5 days of year',
      'units': '%',
      'description': 'Flooding is caused by heavy rainfall, which can lead to rivers overflowing, flash floods and landslides. The amount of rain that falls in a single day can be important, but the amount that falls over a few days can be even more so. This indicator shows the percentage change in amount of rain that falls on the five wettest consecutive days of the year.',
      'periods': {
        'obs': {
          'min': -25,
          'max': 25
        },
        'proj': {
          'min': -10,
          'max': 10
        }
      }
    },
    'txx': {
      'colourscale': 'YlOrRd',
      'dp': 3,
      'title': 'Warmest day of year',
      'units': '°C',
      'description': 'Heatwaves are an increasing threat udner climate change. They can affect the health of people, animals and plants, and can also affect the amount of energy we use to cool our homes and buildings. They can be rather tricky to measure, so instead this indicator shows the change in the temperature of the warmest day of the average year.',
      'periods': {
        'obs': {
          'min': 0,
          'max': 3
        },
        'proj': {
          'min': 0,
          'max': 1
        }
      }
    },
    't': {
      'colourscale': 'YlOrRd',
      'dp': 3,
      'title': 'Temperature',
      'units': '°C',
      'description': 'Changes in temperature are a headline response of climate change. This is because it is easy(ish) to measure and many more consequential impacts track it. This indicator shows the change in the annual temperature, averaged over all the years.',
      'periods': {
        'obs': {
          'min': 0,
          'max': 1
        },
        'proj': {
          'min': 0,
          'max': 1
        }
      }
    }
  };

  // https://colorbrewer2.org
  const colourscales = {
    'Viridis': (min, max) => chroma.scale('Viridis').domain([min, max]),
    'RdYlBu_r': (min, max) => chroma.scale('RdYlBu').domain([max, min]),
    'YlOrRd': (min, max) => chroma.scale('YlOrRd').domain([min, max]),
    'BrBG': (min, max) => chroma.scale('BrBG').domain([min, max]),
    'YlGnBu': (min, max) => chroma.scale('YlGnBu').domain([min, max]),
    'inferno': (min, max) => chroma.scale('inferno').domain([min, max]),
    'cividis': (min, max) => chroma.scale('cividis').domain([min, max]),
  }

  const hex = new ODI.hexmap(
    document.querySelector('.hexmap__outer'),
    {
      // Choose hexjson file to plot
      'hexjson': 'resources/climate_new_constituencies.hexjson',
      'ready': () => {
        // Build dropdown options
        const hexmap_select_var = document.querySelector('[data-hexmap-select-var]');
        for (const key in data_vars) {
          const hexmap_opt = document.createElement('option');
          hexmap_opt.innerText = getLabel(key, data_vars[key]);
          hexmap_opt.value = key;
          hexmap_select_var.appendChild(hexmap_opt);
        }

        const hexmap_select_period = document.querySelector('[data-hexmap-select-period]');
        for (const key in data_periods) {
          const hexmap_opt = document.createElement('option');
          hexmap_opt.innerText = data_periods[key]['title'];
          hexmap_opt.value = key;
          hexmap_select_period.appendChild(hexmap_opt);
        }

        // Load dataset from query parameter
        const params = new URLSearchParams(window.location.search);
        let active_keys = {
          'var': params.get('var'),
          'period': params.get('period')
        }
        if (!active_keys['var'] || !(active_keys['var'] in data_vars)) {
          active_keys['var'] = hexmap_select_var.value;
        }
        if (!active_keys['period'] || !(active_keys['period'] in data_periods)) {
          active_keys['period'] = hexmap_select_period.value;
        }

        // Store some useful attrs for later
        let active_key = active_keys['var']  + '_' + active_keys['period'] ;
        hex.extra = {
          'activeKey': active_key,
          'colourbar': document.querySelector('.hexmap__colourbar'),
          'selectVar': hexmap_select_var,
          'selectPeriod': hexmap_select_period
        };

        // Update hexmap & add auto-update to select change
        hexmap_select_var.value = active_keys['var'];
        updateHexmap(
          hex,
          data_vars,
          data_periods,
          colourscales,
          {
            'period': active_keys['period'],
            'var': active_keys['var']
          }
        )
        hexmap_select_var.addEventListener('change', e => {
          updateHexmap(
            hex,
            data_vars,
            data_periods,
            colourscales,
            {
              'period': hexmap_select_period.value,
              'var': e.target.value
            }
          );
        });
        hexmap_select_period.addEventListener('change', e => {
          updateHexmap(
            hex,
            data_vars,
            data_periods,
            colourscales,
            {
              'period': e.target.value,
              'var': hexmap_select_var.value
            }
          );
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
    if ('mp_firstname' in e.data.data) {
      region_contact =  'MP: ' + e.data.data.mp_firstname +' '+ e.data.data.mp_surname;
      if ('first_party' in e.data.data) {
        region_contact += ' (' + e.data.data.first_party + ')';
      }
    }
    const region_value = getRegionValue(active_key, e.data.data, data_vars);
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

    const key_var = key.split('_').shift();
    let value = data[key];
    let units = '';
    if (key_var in conf) {
      value = getRoundedValue(value, conf[key_var]['dp']);
      if ('units' in conf[key_var]) {
        units = ' ' + conf[key_var]['units'];
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

  function updateHexmap(
    obj,
    all_vars_config,
    all_period_config,
    colourscales,
    keys
  ) {
    const key_var = keys['var'];
    if (!(key_var in all_vars_config)) {
      console.error('Error: ' + key_var + ' not found in config');
      return;
    }

    const key_period = keys['period'];
    if (!(key_period in all_period_config)) {
      console.error('Error: ' + key_period + ' not found in config');
      return;
    }

    // Override variable config with any specific to variable period
    let config = all_vars_config[key_var];
    for (let k in all_vars_config[key_var]['periods'][key_period]) {
      config[k] = all_vars_config[key_var]['periods'][key_period][k];
    }

    const key = keys['var'] + '_' + keys['period'];
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

    // Override with vmin/vmax if defined
    vmin = config['min'] || vmin;
    vmax = config['max'] || vmax;

    // Update description
    const hexmap_select_desc = document.querySelector('[data-hexmap-select-description]');
    hexmap_select_desc.innerText = config.description;

    // NB `colourscale_full` is a chroma.scale object (https://gka.github.io/chroma.js/#color-scales)
    const colourscale_full = colourscales[config.colourscale](vmin, vmax);
    obj.updateColours(r => colourscale_full(obj.mapping.hexes[r][key]));
    obj.extra.colourbar && updateColourbar(obj.extra.colourbar, colourscales, config, key, vmin, vmax);
    obj.extra.activeKey = key;

    // Update url
    const url = new URL(window.location)
    if (url.searchParams.get('var') != key_var) {
      url.searchParams.set('var', key_var);
    }
    if (url.searchParams.get('period') != key_period) {
      url.searchParams.set('period', key_period);
    }
    if (new URL(window.location) !== url) {
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
