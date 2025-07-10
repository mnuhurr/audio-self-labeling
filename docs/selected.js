/*
model-1-head0:
10 crumpling + others
64 bell-like
72 knocking/walking
131 applauding
179 instruments
384 water
797 drawers, opening/closing, inhale
974 thumps

not so obvious:
254 crumpling, kitchen sink, tambourine, ...
*/

// selected classes and their titles/descriptions:
const selected = [
    {
        'id': 10,
        'desc': 'Crumpling, ...'
    },
    {
        'id': 64,
        'desc': 'Bell-like'
    },
    {
        'id': 71,
        'desc': 'Mixed class'
    },
    {
        'id': 72,
        'desc': 'Knocking, walking, ...'
    },
    {
        'id': 131,
        'desc': 'Applauding'
    },
    {
        'id': 179,
        'desc': 'Instruments'
    },
    {
        'id': 384,
        'desc': 'Water'
    },
    {
        'id': 797,
        'desc': 'Opening and closing drawers and doors, inhale, ...'
    },
    {
        'id': 974,
        'desc': 'Thumps etc'
    }
];

function add_player(container, freesound_id) {
    const div = document.createElement('div');
    const iframe = document.createElement('iframe');
    iframe.src = `https://freesound.org/embed/sound/iframe/${freesound_id}/simple/small`;
    iframe.width = "386";
    iframe.height = "31";
    iframe.frameBorder = "0";
    iframe.scrolling = "no";
    div.appendChild(iframe);
    //document.body.appendChild(div);
    container.appendChild(div);
}

function make_players(container, sound_ids) {
    // reset
    sound_ids.forEach(id => {
        add_player(container, id)
    });
}

function update_data(file) {
    const container = document.getElementById("classes");
    fetch(file)
        .then(response => response.json())
        .then(class_data => {
            selected.forEach((data, index) => {
                const cls_data = class_data[data.id];

                // put everythign here
                const cls_div = document.createElement('div');

                const hdr = document.createElement('h3');
                hdr.textContent = `Class ${data.id}: ${data.desc}`;
                cls_div.appendChild(hdr);

                const taglist = document.createElement('div');
                taglist.innerHTML = "<b>tags</b>: " + cls_data.tags.join(', ');
                taglist.style.marginBottom = "10px";
                cls_div.appendChild(taglist);

                const player_container = document.createElement('div');
                make_players(player_container, class_data[data.id].ids);
                cls_div.appendChild(player_container);
                container.appendChild(cls_div);
            });

        })
        .catch(error => console.error('Error loading JSON:', error));
}

update_data("model-1-h0.json");

