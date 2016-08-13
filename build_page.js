$(document).ready(function() {
	// Layout names and gfycat URLs.
    var layouts = [
        ["dwt_72", "EducatedFailingGhostshrimp"],
        ["lesmis", "DangerousEdibleBalloonfish"],
        ["can_96_proper", "GracefulMinorCranefly"],
        ["can_96_poor", "ThoseDrearyIbizanhound"],
        ["rajat11", "CarefreeSmallBaldeagle"],
        ["jazz", "LankyJauntyEsok"],
        ["visbrazil", "SimplisticUncommonIndianelephant"],
        ["grid17", "UncomfortableSolidElephant"],
        ["mesh3e1", "ExcellentFinishedHarborporpoise"],
        ["netscience", "GraciousEverlastingErmine"],
        ["dwt_419", "HandmadeWeirdFlatfish"],
        ["price_1000", "GiftedRegalArgentineruddyduck"],
        ["dwt_1005_proper", "PhysicalEverlastingElectriceel"],
        ["dwt_1005_poor", "GargantuanAlienatedBushsqueaker"],
        ["cage8", "DarkTenderDavidstiger"],
        ["bcsstk09", "LegitimateLittleIndianpalmsquirrel"],
        ["block_2000", "EnlightenedAjarChuckwalla"],
        ["CA-GrQc", "WeeklyPerfumedIndusriverdolphin"],
        ["EVA", "InfiniteFluffyGiraffe"],
        ["us_powergrid", "IllPaltryHorsefly"],
    ]

   	// Build HTML for iframes that have the animations
    for (var i = 0; i < layouts.length; i++) {
    	$(".list").append("<li><a>" + layouts[i][0] + "</a><ul><li></li></ul></li>");
    	layouts[i][1] = "<iframe src='https://gfycat.com/ifr/" + layouts[i][1] + "' frameborder='0' scrolling='no' width='320' height='' allowfullscreen></iframe>";
    }

    // Add layout animation and make it visible when a list item is clicked.
    $('.list > li a').click(function() {
        $(this).parent().find('ul').toggle();
        var idx = $(this).parent().index();
        $(this).parent().find('li').not($('li').has('iframe')).append(layouts[idx][1]);
    });
})
